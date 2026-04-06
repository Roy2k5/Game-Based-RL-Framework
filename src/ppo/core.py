import numpy as np
from enum import Enum
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
import os


class ActionType(Enum):
    DISCRETE = 0
    CONTINUOUS = 1


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    y = np.zeros_like(x)
    running_sum = np.zeros(1, dtype=x.dtype)
    for t in reversed(range(len(x))):
        running_sum = x[t] + discount * running_sum
        y[t] = running_sum
    return y


class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def get_action(self, obs):
        self.eval()
        with torch.inference_mode():
            obs = torch.as_tensor(obs, dtype=torch.float32)
            dis = self._distribution(obs)
            return dis.sample().item()

    def forward(self, obs, act=None):
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)


class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1)


class MLPActorCritic(nn.Module):

    def __init__(
        self,
        observation_dim,
        action_dim,
        device=None,
        action_type=ActionType.DISCRETE,
        hidden_sizes=(32, 32),
        activation=nn.Tanh,
    ):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        obs_dim = observation_dim

        if action_type == ActionType.CONTINUOUS:
            self.pi = MLPGaussianActor(obs_dim, action_dim, hidden_sizes, activation)
        elif action_type == ActionType.DISCRETE:
            self.pi = MLPCategoricalActor(obs_dim, action_dim, hidden_sizes, activation)
        self.v = MLPCritic(obs_dim, hidden_sizes, activation).to(self.device)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy()

    def act(self, obs):
        return self.step(obs)[0]

    def save_checkpoint(self, ckpt_dir, is_best=False):
        pi_filename = "ppo_pi_best.pt" if is_best else "ppo_pi_last.pt"
        v_filename = "ppo_v_best.pt" if is_best else "ppo_v_last.pt"
        pi_filepath = os.path.join(ckpt_dir, pi_filename)
        v_filepath = os.path.join(ckpt_dir, v_filename)
        print(f"[Save checkpoint] checkpoint is saving at {ckpt_dir}")
        torch.save(self.pi.state_dict(), pi_filepath)
        torch.save(self.v.state_dict(), v_filepath)

    def load_checkpoint(self, ckpt_dir, is_best=True):
        pi_filename = "ppo_pi_best.pt" if is_best else "ppo_pi_last.pt"
        v_filename = "ppo_v_best.pt" if is_best else "ppo_v_last.pt"
        pi_filepath = os.path.join(ckpt_dir, pi_filename)
        v_filepath = os.path.join(ckpt_dir, v_filename)
        if not os.path.exists(pi_filepath) or not os.path.exists(v_filepath):
            raise ValueError(f"{pi_filepath} or {v_filepath} does not exist")
        print(f"[Load checkpoint] Checkpoint is loading from {ckpt_dir}!")
        self.pi.load_state_dict(
            torch.load(pi_filepath, map_location=self.device, weights_only=True)
        )
        self.v.load_state_dict(
            torch.load(v_filepath, map_location=self.device, weights_only=True)
        )
