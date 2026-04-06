import torch
from torch import nn
import torch.nn.functional as F
from collections import deque
import numpy as np
import random
import os


class DQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class DQAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr,
        weight_decay,
        start_epsilon,
        final_epsilon,
        epsilon_decay,
        gamma,
        TAU=0.1,
    ):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.start_epsilon = start_epsilon
        self.current_epsilon = start_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.TAU = TAU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.main_model = DQNetwork(state_dim, action_dim).to(self.device)
        self.target_model = DQNetwork(state_dim, action_dim).to(self.device)
        self.target_model.load_state_dict(self.main_model.state_dict())
        self.replay_buffer = deque(maxlen=10000)
        self.optimizer = torch.optim.AdamW(
            self.main_model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    def get_action(self, state, inference_mode=False):
        assert isinstance(state, list), "state must be a list"
        if inference_mode:
            self.main_model.eval()
            with torch.inference_mode():
                state = torch.tensor([state], dtype=torch.float32).to(self.device)
                logits = self.main_model(state)
                return torch.argmax(logits, dim=-1).item()
        if np.random.rand() < self.current_epsilon:
            return np.random.randint(0, self.action_dim)
        else:
            self.main_model.eval()
            with torch.inference_mode():
                state = torch.tensor([state], dtype=torch.float32).to(self.device)
                logits = self.main_model(state)
                return torch.argmax(logits, dim=-1).item()

    def decay_epsilon(self, episode, total_episodes):
        """Linear epsilon decay: from start_epsilon to final_epsilon over total_episodes"""
        self.current_epsilon = self.final_epsilon + (
            self.start_epsilon - self.final_epsilon
        ) * (1 - episode / total_episodes)

    def push(self, state, action, next_state, reward, done):
        self.replay_buffer.append((state, action, next_state, reward, done))

    def sample(self, batch_size):
        batch = random.sample(self.replay_buffer, batch_size)
        state, action, next_state, reward, done = zip(*batch)
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        action = torch.tensor(action, dtype=torch.int64).to(self.device).unsqueeze(1)
        next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float32).to(self.device).unsqueeze(1)
        done = torch.tensor(done, dtype=torch.float32).to(self.device).unsqueeze(1)
        return state, action, next_state, reward, done

    def update_main(self, batch_size):
        self.main_model.train()
        if len(self.replay_buffer) < batch_size:
            return
        state, action, next_state, reward, done = self.sample(batch_size)
        q_value = self.main_model(state).gather(1, action)
        with torch.no_grad():
            next_q_value = self.target_model(next_state).amax(dim=1).unsqueeze(1)
            target_q_value = reward + self.gamma * (1 - done) * next_q_value
        loss = F.mse_loss(q_value, target_q_value)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        new_state_dict = self.target_model.state_dict()
        for key in self.main_model.state_dict():
            new_state_dict[key] = (
                self.TAU * self.main_model.state_dict()[key]
                + (1 - self.TAU) * self.target_model.state_dict()[key]
            )
        self.target_model.load_state_dict(new_state_dict)

    def save_checkpoint(self, ckpt_dir, is_best=False):
        filename = "dqn_best.pt" if is_best else "dqn_last.pt"
        filepath = os.path.join(ckpt_dir, filename)
        print(f"[Load checkpoint] Checkpoint is loading from {filepath}!")
        torch.save(self.main_model.state_dict(), filepath)

    def load_checkpoint(self, ckpt_dir, is_best=True):
        filename = "dqn_best.pt" if is_best else "dqn_last.pt"
        filepath = os.path.join(ckpt_dir, filename)
        if not os.path.exists(filepath):
            raise ValueError(f"{filepath} does not exist")
        print(f"[Save checkpoint] checkpoint is saving at {filepath}")
        self.main_model.load_state_dict(torch.load(filepath, map_location=self.device))
        self.target_model.load_state_dict(self.main_model.state_dict())
