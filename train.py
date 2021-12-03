import argparse
import functools
import os.path
import random
import shutil
import time

import wandb
import png
import glob
import rosbag

from typing import Dict, Any, Callable, List
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

import numpy as np
import pandas as pd
import pyarrow as pa
import onnx # workaround for https://github.com/onnx/onnx/issues/3493
import onnxruntime as rt
import torch

import tensorflow.compat.v1 as tf

from bot_env import SlicedRobotEnvironment, NormalizedRobotEnvironment, RobotEnvironment
from sac import ReplayBuffer, TorchReplayBuffer, SoftActorCritic, TorchLSTMReplayBuffer
import yolo_reward
from split_dropout import SplitDropout
from yolo_reward import get_onnx_prediction
from dump_onnx import export

DEFAULT_MAX_GAP_SECONDS = 5

DEFAULT_PUNISHMENT_MULTIPLIER = 16.0
DEFAULT_MANUAL_DRIVING_REWARD = 1.0

SAMPLES_PER_STEP = 1024*1024

tf.disable_v2_behavior()

@functools.lru_cache()
def get_onnx_sess(onnx_path: str) -> rt.InferenceSession:
    print("Starting ONNX inference session")
    return rt.InferenceSession(onnx_path)


def _flatten(arr):
    return np.reshape(arr, -1)

def read_bag(bag_file: str, backbone_onnx_path: str, reward_func_name: str,
             env: NormalizedRobotEnvironment,
             interpolation_slice: int,
             reward_delay_ms: int,
             punish_backtrack_ms: int) -> pd.DataFrame:
    print(f"Opening {bag_file}")
    bag_cache_name = os.path.join(opt.cache_dir, f"{os.path.basename(bag_file)}_{reward_func_name}_{interpolation_slice}slice+{reward_delay_ms}ms_-{punish_backtrack_ms}ms.arrow")

    try:
        return _read_mmapped_bag(bag_cache_name)
    except IOError:
        write_bag_cache(bag_file, bag_cache_name, backbone_onnx_path, reward_func_name,
                        env=env,
                        interpolation_slice=interpolation_slice,
                        reward_delay_ms=reward_delay_ms,
                        punish_backtrack_ms=punish_backtrack_ms)
        return _read_mmapped_bag(bag_cache_name)


def _read_mmapped_bag(bag_cache_name: str) -> pd.DataFrame:
    source = pa.memory_map(bag_cache_name, "r")
    table = pa.ipc.RecordBatchFileReader(source).read_all()
    return table.to_pandas()


def read_bag_into_numpy(bag_file: str,
                        reward_delay_ms: int,
                        punish_backtrack_ms: int) -> Dict[str, Dict[int, np.ndarray]]:
    bag = rosbag.Bag(bag_file, 'r')
    entries = defaultdict(dict)

    ros_topics = ['/processed_img',
                  '/audio',
                  '/camera/accel/sample',
                  '/camera/gyro/sample',
                  '/head_feedback',
                  '/odrive_feedback',
                  '/cmd_vel',
                  '/head_cmd',
                  '/reward_button',
                  '/reward_button_connected',
                  '/reward_button_override_cmd_vel',
                  '/vbus']

    for topic, msg, ts in bag.read_messages(ros_topics):
        full_ts = ts.nsecs + ts.secs * 1000000000

        if topic == '/processed_img':
            img = []
            for i in range(0, len(msg.data), msg.step):
                img.append(np.frombuffer(msg.data[i:i + msg.step], dtype=np.uint8))

            # Convert list of byte arrays to numpy array
            image_np = np.array(img)
            image_np = image_np.reshape((msg.height, msg.width, -1))
            entries["processed_img"][full_ts] = image_np
        elif topic == '/audio':
            entries["audio"][full_ts] = np.frombuffer(msg.data, dtype=np.float32)
        elif topic == '/reward_button':
            entries["reward_button"][full_ts + punish_backtrack_ms * 1000000] = np.array([msg.data])
        elif topic == '/reward_button_connected':
            entries["reward_button_connected"][full_ts] = np.array([msg.data])
        elif topic == '/reward_button_override_cmd_vel':
            entries["reward_button_override_cmd_vel"][full_ts] = np.array([msg.data])
        elif topic == "/head_feedback":
            entries["head_feedback"][full_ts] = np.array([msg.cur_angle_pitch,
                                                          msg.cur_angle_yaw,
                                                          msg.motor_power_pitch,
                                                          msg.motor_power_yaw])
        elif topic == "/cmd_vel":
            entries["cmd_vel"][full_ts] = np.array([msg.linear.x,
                                                    msg.angular.z])
        elif topic == "/head_cmd":
            entries["head_cmd"][full_ts] = np.array([msg.cmd_angle_pitch,
                                                     msg.cmd_angle_yaw])
        elif topic == "/camera/accel/sample":
            entries["head_accel"][full_ts] = np.array([msg.linear_acceleration.x,
                                                       msg.linear_acceleration.y,
                                                       msg.linear_acceleration.z])
        elif topic == "/camera/gyro/sample":
            entries["head_gyro"][full_ts] = np.array([msg.angular_velocity.x,
                                                      msg.angular_velocity.y,
                                                      msg.angular_velocity.z])
        elif topic == "/odrive_feedback":
            entries["odrive_feedback"][full_ts] = np.array([msg.motor_vel_actual_0,
                                                            msg.motor_vel_actual_1,
                                                            msg.motor_vel_cmd_0,
                                                            msg.motor_vel_cmd_1,
                                                            msg.motor_current_actual_0,
                                                            msg.motor_current_actual_1])
        elif topic == "/vbus":
            entries["vbus"][full_ts] = np.array([msg.data])
        else:
            raise KeyError("Unexpected rosbag topic")

    return entries


class TimedEntryNotFound(Exception):
    pass


def get_timed_entry(entries: Dict, timestamp: int, offset: int) -> np.ndarray:
    if offset == 0:
        if timestamp in entries:
            return entries[timestamp]
        else:
            raise TimedEntryNotFound()
    elif offset < 0:
        keys = sorted([key for key in entries.keys() if key < timestamp], reverse=True)
        offset = abs(offset) - 1
        if offset > len(keys) - 1:
            raise TimedEntryNotFound()

        return entries[keys[offset]]
    else:
        keys = sorted(key for key in entries.keys() if key > timestamp)
        if offset > len(keys):
            raise TimedEntryNotFound()
        return entries[keys[offset - 1]]


@dataclass()
class DatasetEntry:
    ts: int
    observation: np.ndarray
    observation_yolo_intermediate_size: int
    action: np.ndarray
    reward: float
    done: bool


def create_dataset(entries:  Dict[str, Dict[int, np.ndarray]],
                   env: NormalizedRobotEnvironment,
                   backbone_onnx_path: str, reward_func_name: str, interpolation_slice: int) -> List[DatasetEntry]:
    reward_func = getattr(yolo_reward, reward_func_name)
    interpolation_time_master = "processed_img"
    primary = sorted(entries[interpolation_time_master].items())
    processed_entries = []

    for ts, image_np in primary:
        bboxes, intermediate = get_onnx_prediction(get_onnx_sess(backbone_onnx_path), image_np)
        yolo_reward_value = reward_func(bboxes)

        if np.isnan(intermediate).any():
            raise ValueError("Received NaN in yolo reward calculation")

        yolo_intermediate = _flatten(intermediate)[::interpolation_slice]

        try:
            last_reward_button_connection = get_timed_entry(entries["reward_button_connected"], ts, -1)
            last_reward_button_override_cmd_vel = get_timed_entry(entries["reward_button_override_cmd_vel"], ts, -1)
            last_reward_button = get_timed_entry(entries["reward_button"], ts, -1)

            last_head_feedback = get_timed_entry(entries["head_feedback"], ts, -1)
            last_odrive_feedback = get_timed_entry(entries["odrive_feedback"], ts, -1)
            last_vbus = get_timed_entry(entries["vbus"], ts, -1)
            last_head_gyro = get_timed_entry(entries["head_gyro"], ts, -1)
            last_head_accel = get_timed_entry(entries["head_accel"], ts, -1)

            next_head_cmd = get_timed_entry(entries["head_cmd"], ts, 1)
            next_cmd_vel = get_timed_entry(entries["cmd_vel"], ts, 1)
        except TimedEntryNotFound:
            continue

        if not last_reward_button_connection[0]:
            print(f"Skipping entry {ts} because reward button app was not connected")
            continue

        final_reward = yolo_reward_value * opt.base_reward_scale

        move_penalty = abs(next_cmd_vel).mean() * 0
        final_reward -= move_penalty

        look_penalty = abs(next_head_cmd).mean() * 0.002
        final_reward -= look_penalty

        override_reward = DEFAULT_MANUAL_DRIVING_REWARD if last_reward_button_override_cmd_vel else 0.0
        final_reward += override_reward

        final_reward += last_reward_button[0] * DEFAULT_PUNISHMENT_MULTIPLIER

        observation = np.concatenate([
            [env.normalize_pan(last_head_feedback[1]),
             env.normalize_tilt(last_head_feedback[0])],
            last_head_gyro / 10.0,  # Divide radians/sec by ten to center around 0 closer
            last_head_accel / 10.0,  # Divide m/s by 10
            last_odrive_feedback[0:2],  # Only the actual vel, not the commanded vel
            last_vbus - 14.0,  # Volts different from ~50% charge
            yolo_intermediate])

        action = np.concatenate([
            next_cmd_vel,
            [env.normalize_pan(next_head_cmd[1]),
             env.normalize_tilt(next_head_cmd[0])],
        ])

        processed_entries.append(DatasetEntry(
            ts=ts,
            observation=observation,
            observation_yolo_intermediate_size=yolo_intermediate.shape[0],
            reward=final_reward,
            action=action,
            done=False
        ))

    # Set the last entry done flag to true
    if len(processed_entries) > 0:
        processed_entries[-1].done = True
    else:
        print("WARNING: Bag file contained no valid entries")

    return processed_entries


def write_bag_cache(bag_file: str, bag_cache_path: str, backbone_onnx_path: str, reward_func_name: str,
                    env: NormalizedRobotEnvironment,
                    interpolation_slice: int,
                    reward_delay_ms: int,
                    punish_backtrack_ms: int):

    # Read all of the relevant data inside a bag file into a bunch of cleaned up numpy arrays
    bag_entries = read_bag_into_numpy(bag_file,
                                      reward_delay_ms=reward_delay_ms,
                                      punish_backtrack_ms=punish_backtrack_ms)

    # Convert those numpy arrays into a consistent observation, reward, and action vector
    dataset_entries = create_dataset(bag_entries,
                                     env=env,
                                     backbone_onnx_path=backbone_onnx_path,
                                     reward_func_name=reward_func_name,
                                     interpolation_slice=interpolation_slice)


    # Save those into a Dataframe
    df = pd.DataFrame.from_records([asdict(ds) for ds in dataset_entries],
                                    columns=["ts", "observation", "observation_yolo_intermediate_size", "action", "reward", "done"], index="ts")

    # Convert from pandas to Arrow
    table = pa.Table.from_pandas(df)

    # Write out to file
    with pa.OSFile(bag_cache_path, 'wb') as sink:
        with pa.RecordBatchFileWriter(sink, table.schema) as writer:
            writer.write_table(table)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag-dir', type=str, help='directory with bag files to use for training data')
    parser.add_argument("--onnx", type=str, default='./yolov5s_v5_0_op11_rossac.onnx', help='onnx weights path for intermediate stage')
    parser.add_argument("--reward", default='prioritize_centered_spoons_with_nms')
    parser.add_argument('--max-gap', type=int, default=DEFAULT_MAX_GAP_SECONDS, help='max gap in seconds')
    parser.add_argument('--batch-size', type=int, default=128, help='number of samples per training step')
    parser.add_argument('--max-samples', type=int, default=20000, help='max number of training samples to load at once')
    parser.add_argument('--cpu', default=False, action="store_true", help='run training on CPU only')
    parser.add_argument('--reward-delay-ms', type=int, default=100, help='delay reward from action by the specified amount of milliseconds')
    parser.add_argument('--base-reward-scale', type=float, default=1.0, help='Default scaling for the base yolo-reward')
    parser.add_argument('--punish-backtrack-ms', type=int, default=4000, help='backtrack punishment by the button by the specified amount of milliseconds')
    # default rate for dropout assumes small inputs (order of 1024 elements)
    parser.add_argument('--dropout', type=float, default=0.88, help='input dropout rate for training')
    parser.add_argument('--backbone-slice', type=int, default=None, help='use every nth datapoint of the backbone')
    parser.add_argument('--cache-dir', type=str, default=None, help='directory to store precomputed values')
    parser.add_argument('--epoch-steps', type=int, default=100, help='how often to save checkpoints')
    parser.add_argument('--seed', type=int, default=None, help='training seed')
    parser.add_argument('--max-lookback', type=int, default=5, help='max amount of prior steps to feed into a network history')
    parser.add_argument('--history-indexes', type=str, default='-1,-2,-3,-5', help='which indexes to pass into the network')
    parser.add_argument('--gpu-replay-buffer', default=False, action="store_true", help='keep replay buffer in GPU memory')
    parser.add_argument('--no-mixed-precision', default=False, action="store_true", help='use full precision for training')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--lr-critic-schedule', default="lambda step: max(5e-6, 0.9998465 ** step)", help='learning rate schedule (Python lambda) for critic network')
    parser.add_argument('--lr-actor-schedule', default="lambda step: max(5e-6, 0.9998465 ** step)", help='learning rate schedule (Python lambda) for actor network')
    parser.add_argument('--alpha-schedule', default="lambda step: 0.2", help='schedule for entropy regularization (alpha)')
    parser.add_argument('--actor-hidden-sizes', type=str, default='512,256,256', help='actor network hidden layer sizes')
    parser.add_argument('--critic-hidden-sizes', type=str, default='512,256,256', help='critic network hidden layer sizes')
    parser.add_argument('--checkpoint-path', type=str, default='checkpoint/sac.tar', help='path to save/load checkpoint from')
    parser.add_argument('--pretrained-path', type=str, help='path to load pretrained checkpoint, from which the training will be started as a new run')
    parser.add_argument('--wandb-mode', type=str, default='online', help='wandb mode (offline/online/disabled)')
    opt = parser.parse_args()

    if torch.cuda.is_available() and not opt.cpu:
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = None
        print("Using CPU")

    replay_buffer_device = device if opt.gpu_replay_buffer else None

    opt.seed = opt.seed or random.randint(0, 2**31 - 1)
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    backbone_slice = opt.backbone_slice

    cache_dir = opt.cache_dir or os.path.join(opt.bag_dir, "_cache")
    opt.cache_dir = cache_dir
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    start_load = time.perf_counter()
    all_entries = None

    for bag_path in glob.glob(os.path.join(opt.bag_dir, "*.bag")):
        entries = read_bag(bag_path, opt.onnx, opt.reward,
                           env=NormalizedRobotEnvironment(SlicedRobotEnvironment(slice=backbone_slice)),
                           interpolation_slice=backbone_slice,
                           reward_delay_ms=opt.reward_delay_ms,
                           punish_backtrack_ms=opt.punish_backtrack_ms)

        if all_entries is None:
            all_entries = entries
        else:
            all_entries = all_entries.append(entries)

    print(f"Loaded {len(all_entries)} base entries")
    env_fn = lambda: NormalizedRobotEnvironment(slice=backbone_slice)
    replay_buffer_factory = ReplayBuffer
    if opt.max_lookback:
        replay_buffer_factory = lambda obs_dim, act_dim, size: TorchLSTMReplayBuffer(obs_dim=obs_dim, act_dim=act_dim,
                                                                                     size=size, device=replay_buffer_device, history_size=opt.max_lookback)
    else:
        replay_buffer_factory = lambda obs_dim, act_dim, size: TorchReplayBuffer(obs_dim=obs_dim, act_dim=act_dim,
                                                                                 size=size, device=replay_buffer_device)

    actor_hidden_sizes = [int(s) for s in opt.actor_hidden_sizes.split(',')]
    critic_hidden_sizes = [int(s) for s in opt.critic_hidden_sizes.split(',')]
    history_indexes = [int(s) for s in opt.history_indexes.split(',')]

    assert history_indexes[0] == -1, "First history index needs to be -1, and will be replaced with extra_obs during SAC bellman step"

    actor_critic_args = {
        'actor_hidden_sizes': actor_hidden_sizes,
        'critic_hidden_sizes': critic_hidden_sizes,
        'history_indexes': history_indexes,
    }

    env = env_fn()

    example_entry = all_entries.iloc[0]
    backbone_data_size = example_entry.observation_yolo_intermediate_size
    example_observation = example_entry.observation
    dropout = SplitDropout([example_observation.shape[0]-backbone_data_size, backbone_data_size],
                           [0.05, opt.dropout])
    sac = SoftActorCritic(env_fn, replay_size=opt.max_samples, device=device, dropout=dropout,
                          lr=opt.lr,
                          ac_kwargs=actor_critic_args,
                          replay_buffer_factory=replay_buffer_factory)

    resume_dict = sac.load(opt.checkpoint_path) if os.path.exists(opt.checkpoint_path) else None
    pretrained_dict = None
    if opt.pretrained_path:
        if resume_dict:
            raise 'if pretrained-path is specified, checkpoint-path must not already exist'
        pretrained_dict = sac.load(opt.pretrained_path)

    num_samples = min(len(all_entries)-1, opt.max_samples)

    def ts_from_seconds(seconds):
        return int(seconds * 1000000000)

    MAX_TS_DIFF = ts_from_seconds(0.20)

    nans = 0
    oobs = 0
    dones = 0
    lstm_history_count = 0

    for i in tqdm(range(num_samples)):
        entry = all_entries.iloc[i]
        next_entry = all_entries.iloc[i + 1]

        obs = entry.observation
        future_obs = next_entry.observation

        if np.isnan(obs).any() or np.isnan(future_obs).any() or np.isnan(entry.reward).any():
            nans += 1
            continue

        if obs.max() > 1000 or future_obs.max() > 1000:
            oobs += 1
            continue

        if abs(next_entry.name - entry.name) > MAX_TS_DIFF:
            lstm_history_count = 0
            continue

        if entry.done:
            lstm_history_count = 0
            continue

        if next_entry.done:
            dones += 1

        if lstm_history_count >= opt.max_lookback:
            lstm_history_count -= 1

        lstm_history_count += 1
        sac.replay_buffer.store(obs=obs,
                                act=entry.action,
                                rew=entry.reward,
                                next_obs=future_obs,
                                lstm_history_count=lstm_history_count,
                                done=next_entry.done)

    print("filled in replay buffer")
    print(f"Took {time.perf_counter() - start_load}")
    print(f"NaNs in {nans} of {num_samples} samples, large obs in {oobs}")
    print(f"avg. episode len: {(num_samples + 1) / (dones + 1)}")

    wandb.init(project="sac-series1", entity="armyofrobots",
               mode=opt.wandb_mode,
               resume=resume_dict is not None,
               name=resume_dict['run_name'] if resume_dict is not None else None,
               id=resume_dict['run_id'] if resume_dict is not None else None)

     # Save basic params to wandb configuration
    wandb.config.read_dir = opt.bag_dir
    wandb.config.reward_func_name = opt.reward
    if resume_dict and num_samples > wandb.config.num_samples:
        print(f'run upgraded from {wandb.config.num_samples} to {num_samples}, but wandb will not have that information')
    else:
        wandb.config.num_samples = num_samples

    wandb.config.batch_size = opt.batch_size
    wandb.config.gamma = sac.gamma
    wandb.config.polyak = sac.polyak
    wandb.config.lr = sac.lr
    wandb.config.alpha = sac.alpha
    wandb.config.dropout = opt.dropout
    wandb.config.device = str(device)
    wandb.config.reward_delay_ms = opt.reward_delay_ms
    wandb.config.backbone_slice = backbone_slice
    wandb.config.lr_critic_schedule = opt.lr_critic_schedule
    wandb.config.lr_actor_schedule = opt.lr_actor_schedule
    wandb.config.alpha_schedule = opt.alpha_schedule
    wandb.config.actor_hidden_sizes = opt.actor_hidden_sizes
    wandb.config.critic_hidden_sizes = opt.critic_hidden_sizes
    wandb.config.max_lookback = opt.max_lookback
    wandb.config.base_reward_scale = opt.base_reward_scale
    if opt.pretrained_path:
        from pathlib import Path
        wandb.config.pretrained_name = Path(opt.pretrained_path).stem
        print('loaded pretrained: ' + wandb.config.pretrained_name)

    if resume_dict is None:
        wandb.config.seed = opt.seed

    if opt.wandb_mode != 'disabled':
        wandb.watch(sac.ac, log="all", log_freq=100)  # Log gradients periodically

    del all_entries

    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    np.set_printoptions(precision=2)

    steps_per_epoch = opt.epoch_steps
    epoch_start = time.perf_counter()

    i = resume_dict['step']+1 if resume_dict is not None else 0
    if pretrained_dict is not None:
        i = pretrained_dict['step'] + 1
    batches_per_step = round(SAMPLES_PER_STEP / opt.batch_size)

    def lr_scheduler(optim, lambda_code):
        return torch.optim.lr_scheduler.LambdaLR(
            optim,
            # exponential decay; reduces lr by 100x every 30k steps
            lr_lambda=eval(lambda_code),
            last_epoch=i-1)
    pi_lr_schedule = lr_scheduler(sac.pi_optimizer, opt.lr_actor_schedule)
    q_lr_schedule = lr_scheduler(sac.q_optimizer, opt.lr_critic_schedule)

    alpha_schedule = eval(opt.alpha_schedule)

    amp_scaler = torch.cuda.amp.GradScaler() if device and not opt.no_mixed_precision else None

    while True:
        start_time = time.perf_counter()

        pi_lr_schedule.step()
        q_lr_schedule.step()

        alpha = alpha_schedule(i)
        sac.alpha.copy_(torch.tensor(alpha))

        sac.train(batch_size=opt.batch_size, batch_count=batches_per_step, amp_scaler=amp_scaler)
        lossQ = sum(sac.logger.epoch_dict['LossQ'][-batches_per_step:])/batches_per_step
        lossPi = sum(sac.logger.epoch_dict['LossPi'][-batches_per_step:])/batches_per_step

        wandb.log(step=i,
                  data={
                      "LossQ": lossQ,
                      "LossPi": lossPi,
                      "LR_Q": q_lr_schedule.get_last_lr()[0],
                      "LR_Pi": pi_lr_schedule.get_last_lr()[0],
                      "Alpha": alpha,
                  })

        sample_action = sac.logger.epoch_dict['Pi'][-1][0]
        step_time = time.perf_counter() - start_time
        print(f"\r{i:03d} Loss: Q: {lossQ:.4g}, Pi: {lossPi:.4g}. Step time: {step_time:0.3f} Sample action: {sample_action}          ",end="")

        checkpoint_name = f"checkpoints/sac-{wandb.run.name}-{i:05d}"

        epoch_ends = i % steps_per_epoch == 0

        if i % 20 == 0 or epoch_ends:
            with torch.no_grad():
                action_samples, logstd_samples = sac.sample_actions(8)
                action_samples = action_samples.detach().cpu().numpy()
                logstd_samples = logstd_samples.detach().cpu().numpy()

            wandb.log(step=i, data={
                        "action_sample_stdevs": np.mean(np.std(action_samples, axis=0)),
                        "logstds_avg": np.mean(logstd_samples),
                        "pi_grad_l2": max(np.linalg.norm(p.grad.detach().cpu()) for p in sac.ac.pi.parameters()),
                        "q1_grad_l2": max(np.linalg.norm(p.grad.detach().cpu()) for p in sac.ac.q1.parameters()),
                        "q2_grad_l2": max(np.linalg.norm(p.grad.detach().cpu()) for p in sac.ac.q2.parameters()),
                      })

            print()
            print(action_samples)
            print()
            
            samples_name = checkpoint_name + ".samples"
            with open(samples_name, 'w') as samples_file:
                print(f"  LossQ: {lossQ}", file=samples_file)
                print(f"  LossPi: {lossPi}", file=samples_file)
                print(action_samples, file=samples_file)
                print(logstd_samples, file=samples_file)

        if i > 0 and epoch_ends:
            export(sac.ac, device, checkpoint_name + '.onnx', sac.env)
            sac.save(opt.checkpoint_path,
                     run_name=wandb.run.name,
                     run_id=wandb.run.id,
                     step=i,
                     seed=opt.seed)

            shutil.copyfile(src=opt.checkpoint_path, dst=checkpoint_name + '.tar')

            print()
            print("saved " + checkpoint_name)
            print(f"avg. time per step: {(time.perf_counter() - epoch_start)/steps_per_epoch}s")
            print()
            epoch_start = time.perf_counter()
        
        i += 1
        sac.logger.epoch_dict.clear()  # Reset the log buffer, otherwise it would retain data from all steps
