import numpy as np
import scipy.signal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from onnx_normal_dist import ONNXNormal


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def build_obs_history(obs_history, history_indexes, extra_obs=None):
    data = []

    # You can use the extra_obs in an LSTM situation to process "one more frame" at the end of the usual history
    if extra_obs is not None:
        for index in history_indexes[1:]:
            data.append(obs_history[:, obs_history.shape[1] + index + 1, :]) # Offset by one, because you include the extra_obs as the last entry

        data.append(extra_obs)
    else:
        for index in history_indexes:
            data.append(obs_history[:, obs_history.shape[1] + index, :])

    return torch.cat(data, dim=-1)


LOG_STD_MAX = 2
LOG_STD_MIN = -20

class SquashedGaussianMLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_space, history_indexes, deterministic=False, with_logprob=True):
        super().__init__()
        self.net = mlp([obs_dim * len(history_indexes)] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_space = act_space
        self.history_indexes = history_indexes

        self.deterministic = deterministic
        self.with_logprob = with_logprob
        self.with_stddev = False

    def forward(self, obs_history, extra_obs=None):
        if self.with_logprob and self.with_stddev:
            raise NotImplementedError

        # You can use the extra_obs in an LSTM situation to process "one more frame" at the end of the usual history
        final_observation = build_obs_history(obs_history, self.history_indexes, extra_obs)

        net_out = self.net(final_observation)

        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Un comment in order to watch if mu's are collapsing back to constant values
        # mean_mus = mu.mean(axis=0, keepdim=True)
        # diff = (mu - mean_mus).abs().sum()
        # print(f"Mu diff: {diff.detach().cpu()}")

        # Pre-squash distribution and sample
        pi_distribution = ONNXNormal(mu, std)
        if self.deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if self.with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = torch.zeros_like(pi_action)

        pi_action = torch.tanh(pi_action)

        # Now scale each one to the range of the action space
        pi_action = pi_action * torch.from_numpy((self.act_space.high - self.act_space.low) / 2).to(pi_action.device)
        pi_action = pi_action + torch.from_numpy((self.act_space.high + self.act_space.low) / 2).to(pi_action.device)

        if self.with_stddev:
            return pi_action, std
        else:
            return pi_action, logp_pi


class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, history_indexes):
        super().__init__()
        self.history_indexes = history_indexes
        self.q = mlp([obs_dim * len(history_indexes) + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs_history, act, extra_obs=None):
        # You can use the extra_obs in an LSTM situation to process "one more frame" at the end of the usual history
        final_observation = build_obs_history(obs_history, self.history_indexes, extra_obs)

        q = self.q(torch.cat([final_observation, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space,
                 actor_hidden_sizes=(512,256,256),
                 critic_hidden_sizes=(512,256,256),
                 activation=nn.SELU,
                 history_indexes=(-1,)):
        super().__init__()

        if len(actor_hidden_sizes) == 0 or len(critic_hidden_sizes) == 0:
            raise ValueError("Must have at least one hidden layer")

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]

        # build policy and value functions
        self.pi = SquashedGaussianMLPActor(obs_dim, act_dim, actor_hidden_sizes, activation, action_space, history_indexes)
        self.q1 = MLPQFunction(obs_dim, act_dim, critic_hidden_sizes, activation, history_indexes)
        self.q2 = MLPQFunction(obs_dim, act_dim, critic_hidden_sizes, activation, history_indexes)

    def act(self, obs, deterministic=False):
        old_deterministic = self.pi.deterministic
        old_with_logprob = self.pi.with_logprob
        try:
            with torch.no_grad():
                self.pi.deterministic = deterministic
                self.pi.with_logprob = False
                a, _ = self.pi(obs)
                return a.cpu().numpy()
        finally:
            self.pi.deterministic = old_deterministic
            self.pi.with_logprob = old_with_logprob
