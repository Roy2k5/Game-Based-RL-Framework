import numpy as np
import os
import pickle

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


class QTableAgent:
    def __init__(
        self, state_dim, action_dim, gamma, epsilon, epsilon_decay, lr, is_sarsa=True
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.epsilon_decay = epsilon_decay
        self.is_sarsa = is_sarsa
        self.table = {}

    def update(self, state, action, next_state, next_action, reward, done):
        if state not in self.table:
            self.table[state] = np.zeros(self.action_dim)
        if next_state not in self.table:
            self.table[next_state] = np.zeros(self.action_dim)

        if done:
            next_q_value = 0
        else:
            if self.is_sarsa:
                next_q_value = self.table[next_state][next_action]
            else:
                next_q_value = np.max(self.table[next_state])

        self.table[state][action] += self.lr * (
            reward + self.gamma * next_q_value - self.table[state][action]
        )
        self.epsilon *= self.epsilon_decay
        self.epsilon = np.clip(self.epsilon, 0.05, 1)

    def get_action(self, state, is_train=True):
        if state not in self.table:
            self.table[state] = np.zeros(self.action_dim)

        if not is_train:
            return np.argmax(self.table[state])

        if np.random.rand() < self.epsilon:
            return np.random.choice(
                self.action_dim,
                p=np.ones(self.action_dim, dtype=np.float32) / self.action_dim,
            )
        else:
            return np.argmax(self.table[state])

    def save_checkpoint(self, filename):
        checkpoint_path = os.path.join(ROOT_PATH, "checkpoint", "q_table")
        if not os.path.isdir(checkpoint_path):
            os.makedirs(checkpoint_path, exist_ok=True)
        filepath = os.path.join(checkpoint_path, filename)
        with open(filepath, "wb") as file:
            pickle.dump(self, file)

    def load_checkpoint(self, filename):
        filepath = os.path.join(ROOT_PATH, "checkpoint", "q_table", filename)
        with open(filepath, "rb") as file:
            return pickle.load(file)


if __name__ == "__main__":
    q = QTableAgent.load_checkpoint("best.pkl")
    print(q.get_action((3, 4, 5)))
