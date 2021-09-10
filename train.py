import argparse
import os.path
import random
import shutil
import time

import wandb
import glob

from typing import Dict, Any, Callable, List
from tqdm import tqdm

import numpy as np
import torch

import tensorflow.compat.v1 as tf

from bot_env import RobotEnvironment, NormalizedRobotEnvironment
from actor_critic.core import MLPActorCritic
from bag_cache import read_bag
from replay_loader import load_entries, make_observation
from sac import ReplayBuffer, TorchReplayBuffer, SoftActorCritic, TorchLSTMReplayBuffer
from split_dropout import SplitDropout
from dump_onnx import export

DEFAULT_MAX_GAP_SECONDS = 5

DEFAULT_PUNISHMENT_MULTIPLIER = 16

SAMPLES_PER_STEP = 1024*1024

tf.disable_v2_behavior()

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
    parser.add_argument('--lstm-history', type=int, default=4, help='max amount of prior steps to feed into a network history')
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

    loaded_bags = set(glob.glob(os.path.join(opt.bag_dir, "*.bag")))
    for bag_path in loaded_bags:
        entries = read_bag(bag_path,
                           backbone_onnx_path=opt.onnx,
                           cache_dir=opt.cache_dir,
                           camera_topic=opt.camera_topic,
                           reward_func_name=opt.reward,
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
                                                                                     size=size, device=device)
    else:
        replay_buffer_factory = lambda obs_dim, act_dim, size: TorchReplayBuffer(obs_dim=obs_dim, act_dim=act_dim,
                                                                                 size=size, device=device)

    actor_hidden_sizes = [int(s) for s in opt.actor_hidden_sizes.split(',')]
    critic_hidden_sizes =[int(s) for s in opt.critic_hidden_sizes.split(',')]
    actor_critic_args = {
        'actor_hidden_sizes': actor_hidden_sizes,
        'critic_hidden_sizes': critic_hidden_sizes,
    }

    env = env_fn()

    example_entry = all_entries.iloc[0]
    backbone_data_size = example_entry.yolo_intermediate[::backbone_slice].shape[0]
    example_observation = make_observation(env, example_entry, opt.backbone_slice)
    dropout = SplitDropout([example_observation.shape[0]-backbone_data_size, backbone_data_size],
                           [0.05, opt.dropout])
    sac = SoftActorCritic(env_fn, replay_size=opt.max_samples, device=device, dropout=dropout,
                          lr=opt.lr,
                          ac_kwargs=actor_critic_args,
                          replay_buffer_factory=replay_buffer_factory)

    load_entries(all_entries, sac.replay_buffer, env, opt.backbone_slice)

    print("filled in replay buffer")
    print(f"Took {time.perf_counter() - start_load}")

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
    batches_per_step = SAMPLES_PER_STEP // opt.batch_size

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
