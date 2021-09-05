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
from dataclasses import dataclass, field
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

import numpy as np
import pandas as pd
import pyarrow as pa
import torch
import onnxruntime as rt

import tensorflow.compat.v1 as tf

from bot_env import RobotEnvironment, NormalizedRobotEnvironment
from actor_critic.core import MLPActorCritic
from sac import ReplayBuffer, TorchReplayBuffer, SoftActorCritic, TorchLSTMReplayBuffer
import yolo_reward
from split_dropout import SplitDropout
from yolo_reward import get_prediction, get_intermediate_layer
from dump_onnx import export

DEFAULT_MAX_GAP_SECONDS = 5

DEFAULT_PUNISHMENT_MULTIPLIER = 16

SAMPLES_PER_STEP = 1024*1024

tf.disable_v2_behavior()

@functools.lru_cache()
def get_onnx_sess(onnx_path: str) -> rt.InferenceSession:
    print("Starting ONNX inference session")
    return rt.InferenceSession(onnx_path)


def interpolate(pre_ts, pre_data, ts, post_ts, post_data):
    interval_len = post_ts - pre_ts
    if interval_len == 0: return pre_data
    target_offset = ts - pre_ts
    interval_diff = post_data - pre_data
    return pre_data + (interval_diff * target_offset / interval_len)

assert(interpolate(1, 1, 5, 11, 11) == 5)


def interpolate_events(primary, secondaries, max_gap_ns):
    primary = sorted(primary.items())
    secondaries = list(map(lambda dict: list(sorted(dict.items())), secondaries))

    result = []
    for ts, event in primary:
        interpolated = []
        for i in range(len(secondaries)):
            while len(secondaries[i]) > 0:
                if len(secondaries[i]) == 1:
                    # only one potentially correlated reading
                    secondary_ts, data = secondaries[i][0]
                    if abs(secondary_ts - ts) < max_gap_ns:
                        interpolated.append(data)
                    break
                elif secondaries[i][1][0] < ts:
                    # next and next+1 readings are both before "now"
                    # means we can drop next and advance to next+1
                    secondaries[i].pop(0)
                elif secondaries[i][0][0] >= ts:
                    # all readings are after "now", so we may only use the first
                    secondary_ts, data = secondaries[i][0]
                    if abs(secondary_ts - ts) < max_gap_ns:
                        interpolated.append(data)
                    break
                else:
                    # at this point next is before "now", and next+1 is after "now"
                    pre_ts, pre_data = secondaries[i][0]
                    post_ts, post_data = secondaries[i][1]
                    if abs(pre_ts - ts) < max_gap_ns and abs(post_ts - ts) < max_gap_ns:
                        data = interpolate(pre_ts, pre_data, ts, post_ts, post_data)
                        interpolated.append(data)
                    elif abs(pre_ts - ts) < max_gap_ns:
                        interpolated.append(pre_data)
                    elif abs(post_ts - ts) < max_gap_ns:
                        interpolated.append(post_data)
                    break
            if len(interpolated) != i + 1:
                break
        if len(interpolated) != len(secondaries):
            continue
        result.append((ts, event, interpolated))

    return result

def _flatten(arr):
    return np.reshape(arr, -1)


DATAFRAME_COLUMNS = [
    # The first entry is the one that we will interpolate over
    "yolo_intermediate",
    "reward",

    # pan, tilt in steps, (1024 steps = 300 degrees)
    "dynamixel_cur_state",
    "dynamixel_command_state",

    # commanded forward speed, rotational speed
    "cmd_vel",

    # Each wheels actual speed in rotations per second
    "odrive_feedback",

    # head gyro rate, in radians per second
    "head_gyro",

    # head acceleration, in meters per second
    "head_accel",

    # robot bus voltage, in Volts
    "vbus",

    # indicates reward from stop button (0 = unpressed, -1 = pressed)
    "punishment",
]


def read_bag(bag_file: str, backbone_onnx_path: str, reward_func_name: str,
            reward_delay_ms: int, punish_backtrack_ms: int) -> pd.DataFrame:
    print(f"Opening {bag_file}")
    bag_cache_name = os.path.join(opt.cache_dir, f"{os.path.basename(bag_file)}_{reward_func_name}_+{reward_delay_ms}ms_-{punish_backtrack_ms}ms.arrow")

    try:
        return _read_mmapped_bag(bag_cache_name)
    except IOError:
        write_bag_cache(bag_file, bag_cache_name, backbone_onnx_path, reward_func_name,
                        reward_delay_ms=reward_delay_ms, punish_backtrack_ms=punish_backtrack_ms)
        return _read_mmapped_bag(bag_cache_name)


def _read_mmapped_bag(bag_cache_name: str) -> pd.DataFrame:
    source = pa.memory_map(bag_cache_name, "r")
    table = pa.ipc.RecordBatchFileReader(source).read_all()
    return table.to_pandas()


def write_bag_cache(bag_file: str, bag_cache_path: str, backbone_onnx_path: str, reward_func_name: str,
                    reward_delay_ms: int, punish_backtrack_ms: int):
    bag = rosbag.Bag(bag_file, 'r')
    entries = defaultdict(dict)
    reward_func = getattr(yolo_reward, reward_func_name)

    # If you are the first bag in a series, don't output any entries until you received one message from each channel
    # This is because it can take some time for everything to fully initialize (ex. if building tensorrt models),
    # and we don't want bogus data points.
    wait_for_each_msg = bag_file.endswith("_0.bag")
    received_topic = defaultdict(bool)
    ros_topics = [opt.camera_topic,
                  '/dynamixel_workbench/dynamixel_state',
                  '/camera/accel/sample',
                  '/camera/gyro/sample',
                  '/head_feedback',
                  '/cmd_vel',
                  '/odrive_feedback',
                  '/reward_button',
                  '/vbus']

    for topic, msg, ts in bag.read_messages(ros_topics):
        full_ts = ts.nsecs + ts.secs * 1000000000

        received_topic[topic] = True
        if wait_for_each_msg and not all(received_topic[topic] for topic in ros_topics):
            continue

        if topic == opt.camera_topic:
            img = []
            for i in range(0, len(msg.data), msg.step):
                img.append(np.frombuffer(msg.data[i:i + msg.step], dtype=np.uint8))

            assert "infra" in opt.camera_topic, "Expecting mono infrared images only right now"

            # Convert list of byte arrays to numpy array
            image_np = np.array(img)
            pred = get_prediction(get_onnx_sess(backbone_onnx_path), image_np)
            intermediate = get_intermediate_layer(pred)
            reward = reward_func(pred)

            if np.isnan(intermediate).any():
                print(f"Invalid YOLO output in bag: {bag_file} at ts {ts}")
                img_mode = 'L' if "infra" in opt.camera_topic else 'RGB'
                png.from_array(img, mode=img_mode).save(os.path.join(opt.cache_dir, f"error_{os.path.basename(bag_file)}_{ts}.png"))
                continue

            entries["yolo_intermediate"][full_ts] = _flatten(intermediate)
            entries["reward"][full_ts + reward_delay_ms * 1000000] = reward
        elif topic == '/reward_button':
            entries["punishment"][full_ts + punish_backtrack_ms * 1000000] = np.array([msg.data])
        elif topic == '/dynamixel_workbench/dynamixel_state':
            entries["dynamixel_cur_state"][full_ts] = np.array([msg.dynamixel_state[0].present_position,
                                                                msg.dynamixel_state[1].present_position])
        elif topic == "/head_feedback":
            entries["dynamixel_command_state"][full_ts] = np.array([msg.pan_command,
                                                                    msg.tilt_command])
        elif topic == "/cmd_vel":
            entries["cmd_vel"][full_ts] = np.array([msg.linear.x,
                                                    msg.angular.z])
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
                                                            msg.motor_vel_cmd_1])
        elif topic == "/vbus":
            entries["vbus"][full_ts] = np.array([msg.data])
        else:
            raise KeyError("Unexpected rosbag topic")

    interpolated = interpolate_events(entries["yolo_intermediate"], [entries[key] for key in DATAFRAME_COLUMNS[1:]],
                                      max_gap_ns=1000 * 1000 * 1000)

    df = pd.DataFrame.from_records([[ts, event, *interps] for (ts, event, interps) in interpolated],
                                    columns=["ts", *DATAFRAME_COLUMNS], index="ts")

    # Convert from pandas to Arrow
    table = pa.Table.from_pandas(df)

    # Write out to file
    with pa.OSFile(bag_cache_path, 'wb') as sink:
        with pa.RecordBatchFileWriter(sink, table.schema) as writer:
            writer.write_table(table)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag-dir', type=str, help='directory with bag files to use for training data')
    parser.add_argument("--onnx", type=str, default='./yolov5s.onnx', help='onnx weights path for intermediate stage')
    parser.add_argument("--camera_topic", default='/camera/infra2/image_rect_raw')
    parser.add_argument("--reward", default='sum_centered_objects_present')
    parser.add_argument('--max-gap', type=int, default=DEFAULT_MAX_GAP_SECONDS, help='max gap in seconds')
    parser.add_argument('--batch-size', type=int, default=128, help='number of samples per training step')
    parser.add_argument('--max-samples', type=int, default=20000, help='max number of training samples to load at once')
    parser.add_argument('--cpu', default=False, action="store_true", help='run training on CPU only')
    parser.add_argument('--reward-delay-ms', type=int, default=100, help='delay reward from action by the specified amount of milliseconds')
    parser.add_argument('--punish-backtrack-ms', type=int, default=4000, help='backtrack punishment by the button by the specified amount of milliseconds')
    # default rate for dropout assumes small inputs (order of 1024 elements)
    parser.add_argument('--dropout', type=float, default=0.88, help='input dropout rate for training')
    parser.add_argument('--backbone-slice', type=int, default=None, help='use every nth datapoint of the backbone')
    parser.add_argument('--cache-dir', type=str, default=None, help='directory to store precomputed values')
    parser.add_argument('--epoch-steps', type=int, default=100, help='how often to save checkpoints')
    parser.add_argument('--seed', type=int, default=None, help='training seed')
    parser.add_argument('--lstm-history', type=int, default=240, help='max amount of prior steps to feed into a network history')
    parser.add_argument('--history-indexes', type=str, default='-1,-2,-3,-5,-8,-13,-21,-34,-55,-89,-144,-233',
                        help='which indexes to pass into the network')
    parser.add_argument('--gpu-replay-buffer', default=False, action="store_true", help='keep replay buffer in GPU memory')
    parser.add_argument('--no-mixed-precision', default=False, action="store_true", help='use full precision for training')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--lr-critic-schedule', default="lambda step: max(5e-6, 0.9998465 ** step)", help='learning rate schedule (Python lambda) for critic network')
    parser.add_argument('--lr-actor-schedule', default="lambda step: max(5e-6, 0.9998465 ** step)", help='learning rate schedule (Python lambda) for actor network')
    parser.add_argument('--alpha-schedule', default="lambda step: 0.2", help='schedule for entropy regularization (alpha)')
    parser.add_argument('--actor-hidden-sizes', type=str, default='512,256,256', help='actor network hidden layer sizes')
    parser.add_argument('--critic-hidden-sizes', type=str, default='512,256,256', help='critic network hidden layer sizes')
    parser.add_argument('--checkpoint-path', type=str, default='checkpoint/sac.tar', help='path to save/load checkpoint from')
    parser.add_argument('--wandb-mode', type=str, default='online', help='wandb mode (offline/online/disabled)')
    opt = parser.parse_args()

    if torch.cuda.is_available() and not opt.cpu:
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = None
        print("Using CPU")

    opt.seed = opt.seed or random.randint(0, 2**31 - 1)
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    cache_dir = opt.cache_dir or os.path.join(opt.bag_dir, "_cache")
    opt.cache_dir = cache_dir
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    start_load = time.perf_counter()
    all_entries = None

    for bag_path in glob.glob(os.path.join(opt.bag_dir, "*.bag")):
        entries = read_bag(bag_path, opt.onnx, opt.reward,
                           reward_delay_ms=opt.reward_delay_ms,
                           punish_backtrack_ms=opt.punish_backtrack_ms)

        if all_entries is None:
            all_entries = entries
        else:
            all_entries = all_entries.append(entries)

    print(f"Loaded {len(all_entries)} base entries")

    backbone_slice = opt.backbone_slice
    env_fn = lambda: NormalizedRobotEnvironment(slice=backbone_slice)
    replay_buffer_factory = ReplayBuffer
    if opt.lstm_history:
        replay_buffer_factory = lambda obs_dim, act_dim, size: TorchLSTMReplayBuffer(obs_dim=obs_dim, act_dim=act_dim,
                                                                                     size=size, device=device, history_size=opt.lstm_history)
    else:
        replay_buffer_factory = lambda obs_dim, act_dim, size: TorchReplayBuffer(obs_dim=obs_dim, act_dim=act_dim,
                                                                                 size=size, device=device)

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

    def normalize_pantilt(pantilt):
        return env.normalize_pan(pantilt[0, np.newaxis]), env.normalize_tilt(pantilt[1, np.newaxis])

    def make_observation(interpolated_entry):
        pan_curr, tilt_curr = normalize_pantilt(interpolated_entry.dynamixel_cur_state)
        return np.concatenate([pan_curr, tilt_curr,
                               interpolated_entry.head_gyro / 10.0,  # Divide radians/sec by ten to center around 0 closer
                               interpolated_entry.head_accel / 10.0,  # Divide m/s by 10, and center the y axis
                               interpolated_entry.odrive_feedback[0:2],  # Only the actual vel, not the commanded vel
                               interpolated_entry.vbus - 27.0,  # Volts different from ~50% charge
                               interpolated_entry.yolo_intermediate[::backbone_slice]])

    example_entry = all_entries.iloc[0]
    backbone_data_size = example_entry.yolo_intermediate[::backbone_slice].shape[0]
    example_observation = make_observation(example_entry)
    dropout = SplitDropout([example_observation.shape[0]-backbone_data_size, backbone_data_size],
                           [0.05, opt.dropout])
    sac = SoftActorCritic(env_fn, replay_size=opt.max_samples, device=device, dropout=dropout,
                          lr=opt.lr,
                          ac_kwargs=actor_critic_args,
                          replay_buffer_factory=replay_buffer_factory)

    num_samples = min(len(all_entries)-1, opt.max_samples)

    used = [False for _ in range(num_samples + 1)]

    nans = 0
    oobs = 0
    dones = 0

    def ts_from_seconds(seconds):
        return int(seconds * 1000000000)

    MIN_TS_DIFF = ts_from_seconds(0.47)
    MAX_TS_DIFF = ts_from_seconds(0.75)

    t = tqdm(total=num_samples)
    loaded = 0

    threads = 0
    while not all(used[:-1]):
        threads += 1
        last_terminated = False
        lstm_history_count = 0
        last_ts = None
        i = 0
        while i < num_samples:
            if used[i]:
                i += 1
                continue

            used[i] = True
            loaded += 1
            t.update()

            entry = all_entries.iloc[i]
            ts = entry.name
            i += 1
            while i < num_samples and (all_entries.iloc[i].name < ts + MIN_TS_DIFF or used[i]):
                i += 1
            if i >= num_samples:
                continue
            next_entry = all_entries.iloc[i]
            if next_entry.name >= ts + MAX_TS_DIFF:
                lstm_history_count = 0
                continue

            if lstm_history_count >= opt.lstm_history:
                lstm_history_count -= 1

            pan_command, tilt_command = normalize_pantilt(entry.dynamixel_command_state)
            pan_curr, tilt_curr = normalize_pantilt(entry.dynamixel_cur_state)

            move_penalty = abs(entry.cmd_vel).mean() * 0.002
            pantilt_penalty = float((abs(pan_command - pan_curr) + abs(tilt_command - tilt_curr)) * 0.001)
            if move_penalty + pantilt_penalty > 10:
                print("WARNING: high move penalty!")
            reward = entry.reward
            reward -= move_penalty + pantilt_penalty
            reward += next_entry.punishment * DEFAULT_PUNISHMENT_MULTIPLIER

            obs = make_observation(entry)
            future_obs = make_observation(next_entry)

            if np.isnan(obs).any() or np.isnan(future_obs).any() or np.isnan(reward).any():
                nans += 1
                continue

            if obs.max() > 1000 or future_obs.max() > 1000:
                oobs += 1
                continue

            terminated = next_entry.punishment < -0.0
            if terminated and last_terminated:
                continue
            last_terminated = terminated
            if terminated:
                dones += 1

            lstm_history_count += 1
            sac.replay_buffer.store(obs=obs,
                act=np.concatenate([entry.cmd_vel, pan_command, tilt_command]),
                rew=reward,
                next_obs=future_obs,
                lstm_history_count=lstm_history_count,
                done=terminated)

    t.close()

    print("filled in replay buffer")
    print(f"Took {time.perf_counter() - start_load}")
    print(f"NaNs in {nans} of {num_samples} samples, large obs in {oobs}, threads: {threads}")
    print(f"avg. episode len: {(num_samples + 1) / (dones + 1)}")

    resume_dict = sac.load(opt.checkpoint_path) if os.path.exists(opt.checkpoint_path) else None

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
    wandb.config.lstm_history = opt.lstm_history
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
