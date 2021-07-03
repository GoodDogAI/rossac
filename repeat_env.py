from gym.spaces import Box
import numpy as np
import random
import torch

from actor_critic.core import MLPActorCritic
from sac import SoftActorCritic

class RepeatObservationEnv:
    action_space = observation_space = Box(low=-1.0, high=+1.0, shape=(1,), dtype=np.float32)

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    RANDOM_SAMPLE_COUNT = 8*1024
    BATCH_SIZE=128

    sac = SoftActorCritic(RepeatObservationEnv, replay_size=RANDOM_SAMPLE_COUNT, device=device)

    obs = random.uniform(-1, +1)
    for i in range(RANDOM_SAMPLE_COUNT):
        act = random.uniform(-1, +1)
        reward = 2 - abs(act-obs)
        new_obs = random.uniform(-1, +1)
        done = False # or (i-1) % MAX_STEPS == 0
        
        sac.replay_buffer.store(obs=obs, act=act, rew=reward, next_obs=new_obs, done=done)
        
        obs = new_obs

    for i in range(1000):
        sac.train(batch_size=BATCH_SIZE, batch_count=32)
        print(f"  LossQ: {sum(sac.logger.epoch_dict['LossQ'][-BATCH_SIZE:])/BATCH_SIZE}", end=None)
        print(f"  LossPi: {sum(sac.logger.epoch_dict['LossPi'][-BATCH_SIZE:])/BATCH_SIZE}", end=None)
        sample_action = sac.logger.epoch_dict['Pi'][-1][0]
        print(f"  Sample Action: {sample_action[0]:.2f}")

        if i % 20 == 19:
            action_samples = sac.sample_actions(8).cpu()
            print(action_samples)
