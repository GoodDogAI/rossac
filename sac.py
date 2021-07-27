from copy import deepcopy
import itertools
import numpy as np
import os
import torch
from torch.nn import Dropout
from torch.optim import Adam
import gym
import time
import actor_critic.core as core
from spinup.utils.logx import EpochLogger


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}

class TorchReplayBuffer:
    def __init__(self, obs_dim, act_dim, size, device=None):
        self.device = device
        self.obs_buf = torch.zeros(core.combined_shape(size, obs_dim), device=device, dtype=torch.float32)
        self.obs2_buf = torch.zeros(core.combined_shape(size, obs_dim), device=device, dtype=torch.float32)
        self.act_buf = torch.zeros(core.combined_shape(size, act_dim), device=device, dtype=torch.float32)
        self.rew_buf = torch.zeros(size, device=device, dtype=torch.float32)
        self.done_buf = torch.zeros(size, device=device, dtype=torch.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        with torch.no_grad():
            # replace random entry when full
            if self.size == self.max_size:
                self.ptr = torch.randint(0, self.max_size, (), dtype=torch.int64, device=self.device)
            self.obs_buf[self.ptr] = torch.as_tensor(obs, device=self.device)
            self.obs2_buf[self.ptr] = torch.as_tensor(next_obs, device=self.device)
            self.act_buf[self.ptr] = torch.as_tensor(act, device=self.device)
            self.rew_buf[self.ptr] = torch.as_tensor(rew, device=self.device)
            self.done_buf[self.ptr] = torch.as_tensor(done, device=self.device)
            self.ptr += 1
            self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = torch.randint(0, self.size, (batch_size,), dtype=torch.int64, device=self.device)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return batch

class SoftActorCritic:
    def __init__(self, env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(),
            replay_size=int(1e6), gamma=0.99,
            polyak=0.995, lr=1e-3, alpha=0.2,
            max_ep_len=1000,
            device=None,
            dropout=0.88,
            replay_buffer_factory=ReplayBuffer,
            logger_kwargs=dict()):
        """
        Soft Actor-Critic (SAC)


        Args:
            env_fn : A function which creates a copy of the environment.
                The environment must satisfy the OpenAI Gym API.

            actor_critic: The constructor method for a PyTorch Module with an ``act``
                method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
                The ``act`` method and ``pi`` module should accept batches of
                observations as inputs, and ``q1`` and ``q2`` should accept a batch
                of observations and a batch of actions as inputs. When called,
                ``act``, ``q1``, and ``q2`` should return:

                ===========  ================  ======================================
                Call         Output Shape      Description
                ===========  ================  ======================================
                ``act``      (batch, act_dim)  | Numpy array of actions for each
                                            | observation.
                ``q1``       (batch,)          | Tensor containing one current estimate
                                            | of Q* for the provided observations
                                            | and actions. (Critical: make sure to
                                            | flatten this!)
                ``q2``       (batch,)          | Tensor containing the other current
                                            | estimate of Q* for the provided observations
                                            | and actions. (Critical: make sure to
                                            | flatten this!)
                ===========  ================  ======================================

                Calling ``pi`` should return:

                ===========  ================  ======================================
                Symbol       Shape             Description
                ===========  ================  ======================================
                ``a``        (batch, act_dim)  | Tensor containing actions from policy
                                            | given observations.
                ``logp_pi``  (batch,)          | Tensor containing log probabilities of
                                            | actions in ``a``. Importantly: gradients
                                            | should be able to flow back into ``a``.
                ===========  ================  ======================================

            ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object
                you provided to SAC.

            replay_size (int): Maximum length of replay buffer.

            gamma (float): Discount factor. (Always between 0 and 1.)

            polyak (float): Interpolation factor in polyak averaging for target
                networks. Target networks are updated towards main networks
                according to:

                .. math:: \\theta_{\\text{targ}} \\leftarrow
                    \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

                where :math:`\\rho` is polyak. (Always between 0 and 1, usually
                close to 1.)

            lr (float): Learning rate (used for both policy and value learning).

            alpha (float): Entropy regularization coefficient. (Equivalent to
                inverse of reward scale in the original SAC paper.)

            max_ep_len (int): Maximum length of trajectory / episode / rollout.

            logger_kwargs (dict): Keyword args for EpochLogger.
        """

        self.logger = EpochLogger(**logger_kwargs)
        self.logger.save_config(locals())

        self.max_ep_len = max_ep_len

        self.alpha = alpha
        self.gamma = gamma
        self.polyak = polyak
        self.lr = lr

        self.device = device

        self.env, self.test_env = env_fn(), env_fn()
        obs_dim = self.env.observation_space.shape
        act_dim = self.env.action_space.shape[0]

        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        act_limit = self.env.action_space.high[0]

        # Create actor-critic module and target networks
        self.ac = actor_critic(self.env.observation_space, self.env.action_space, **ac_kwargs)

        self.ac_targ = deepcopy(self.ac)
        self.ac = self._to_device(self.ac)
        self.ac_targ = self._to_device(self.ac_targ)

        self._freeze_target()

        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())

        # Experience buffer
        self.replay_buffer = replay_buffer_factory(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

        # Count variables (protip: try to get a feel for how different size networks behave!)
        var_counts = tuple(core.count_vars(module) for module in [self.ac.pi, self.ac.q1, self.ac.q2])
        self.logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n' % var_counts)

        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=lr)
        self.q_optimizer = Adam(self.q_params, lr=lr)

        self.dropout = Dropout(p=dropout)

    # Set up function for computing SAC Q-losses
    def compute_loss_q(self, data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = self.ac.q1(o, a)
        q2 = self.ac.q2(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.ac.pi(o2)

            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(o2, a2)
            q2_pi_targ = self.ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().cpu().numpy(),
                      Q2Vals=q2.detach().cpu().numpy())

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(self, data):
        o = data['obs']
        pi, logp_pi = self.ac.pi(o)
        q1_pi = self.ac.q1(o, pi)
        q2_pi = self.ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().cpu().numpy(),
                       Pi=pi.detach().cpu().numpy())

        return loss_pi, pi_info

    def update(self, data):
        # Take both observations to the GPU for faster training
        for key in data.keys():
            data[key] = data[key].to(device=self.device)

        data['obs'] = self.dropout(data['obs'])
        data['obs2'] = self.dropout(data['obs2'])

        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q, q_info = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        # Record things
        self.logger.store(LossQ=loss_q.item(), **q_info)

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in self.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params:
            p.requires_grad = True

        # Record things
        self.logger.store(LossPi=loss_pi.item(), **pi_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    def get_action(self, observation, deterministic=False):
        return self.ac.act(torch.as_tensor(observation, dtype=torch.float32, device=self.device),
                           deterministic)

    def test_agent(self, episode_count):
        avg_rew = 0
        avg_action = 0
        step_count = 0
        for j in range(episode_count):
            o, d, ep_ret, ep_len = self.test_env.reset(), False, 0, 0
            while not (d or (ep_len == self.max_ep_len)):
                # Take deterministic actions at test time
                action = self.get_action(o, True)
                step_count += 1
                avg_action += abs(action)
                o, r, d, _ = self.test_env.step(action)
                ep_ret += r
                ep_len += 1
            avg_rew += ep_ret
            self.logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)
        avg_rew /= episode_count
        avg_action /= step_count
        return avg_rew, avg_action

    def train(self, batch_size, batch_count):
        for j in range(batch_count):
            batch = self.replay_buffer.sample_batch(batch_size)
            self.update(data=batch)

    def sample_actions(self, batch_size):
        batch = self.replay_buffer.sample_batch(batch_size)
        o = batch['obs'].to(device=self.device)
        det = self.ac.pi.deterministic
        try:
            self.ac.pi.deterministic = True
            return self.ac.pi(o)
        finally:
            self.ac.pi.deterministic = det

    def collect_observations(self, steps, use_brain=True):
        ep_len, ep_ret = 0, 0
        o = self.env.reset()
        for t in range(steps):
            # Until start_steps have elapsed, randomly sample actions
            # from a uniform distribution for better exploration. Afterwards,
            # use the learned policy.
            if use_brain:
                a = self.get_action(o)
            else:
                a = self.env.action_space.sample()

            # Step the env
            o2, r, d, _ = self.env.step(a)
            ep_ret += r
            ep_len += 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            d = False if ep_len == self.max_ep_len else d

            # Store experience to replay buffer
            self.replay_buffer.store(o, a, r, o2, d)

            # Super critical, easy to overlook step: make sure to update
            # most recent observation!
            o = o2

            # End of trajectory handling
            if d or (ep_len == self.max_ep_len):
                o, ep_ret, ep_len = self.env.reset(), 0, 0

    def _to_device(self, module):
        if self.device:
            return module.to(self.device)
        return module

    def _freeze_target(self):
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False

    def save(self, path, **kwargs):
        dict = {
            'ac': self.ac.state_dict(),
            'ac_targ': self.ac_targ.state_dict(),
            'pi_opt': self.pi_optimizer.state_dict(),
            'q_opt': self.q_optimizer.state_dict(),
        }
        dict.update(kwargs)
        torch.save(dict, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.ac.load_state_dict(checkpoint['ac'])
        self.ac.train()
        self.ac_targ.load_state_dict(checkpoint['ac_targ'])
        self.ac_targ.train()
        self._freeze_target()
        self.pi_optimizer.load_state_dict(checkpoint['pi_opt'])
        self.q_optimizer.load_state_dict(checkpoint['q_opt'])
        return checkpoint



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='MountainCarContinuous-v0')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='sac')
    parser.add_argument('--cpu', default=False, action='store_true')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    torch.set_num_threads(torch.get_num_threads())

    batches_per_step = 10
    batch_size = 100

    if args.seed:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    replay_buffer_factory = ReplayBuffer
    if torch.cuda.is_available() and not args.cpu:
        device = torch.device("cuda")
        replay_buffer_factory = lambda obs_dim, act_dim, size: TorchReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=size, device=device)
        print("Using CUDA")
    else:
        device = None
        print("Using CPU")

    sac = SoftActorCritic(lambda: gym.make(args.env), actor_critic=core.MLPActorCritic,
        device=device, replay_buffer_factory=replay_buffer_factory,
        ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
        gamma=args.gamma,
        logger_kwargs=logger_kwargs)
    print('collecting observations from random actions...')
    sac.collect_observations(steps=10000, use_brain=False)
    print('initial training...')
    sac.train(batch_size=batch_size, batch_count=10)
    print('')
    for epoch in range(0, args.epochs):
        print('epoch {}: playing around...'.format(epoch))
        sac.collect_observations(steps=1000, use_brain=True)
        print('epoch {}: grokking...'.format(epoch))
        
        sac.train(batch_size=batch_size, batch_count=batches_per_step)

        lossQ = sum(sac.logger.epoch_dict['LossQ'][-batches_per_step:])/batches_per_step
        lossPi = sum(sac.logger.epoch_dict['LossPi'][-batches_per_step:])/batches_per_step
        sample_action = sac.logger.epoch_dict['Pi'][-1][0]

        print(f"{epoch:03d} Loss: Q: {lossQ:.4g}, Pi: {lossPi:.4g}. Sample action: {sample_action}          ")

        print('self-testing...', end='')
        avg_rew, avg_act = sac.test_agent(episode_count=10)
        print(' avg. reward: {} avg. abs action: {}\n'.format(avg_rew, avg_act))
