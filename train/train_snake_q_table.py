from env.snake.snake import *
from src.q_learning.q_table import QTableAgent
import pygame
import random
import numpy as np
import os


class RandomPolicy:
    """Fallback policy for model mode demo."""

    def predict(self, state: np.ndarray) -> int:
        _ = state
        return random.randint(0, 2)


def _is_exit_key(event: Any) -> bool:
    if event.type != pygame.KEYDOWN:
        return False

    if event.key in (pygame.K_q, pygame.K_ESCAPE):
        return True

    return getattr(event, "unicode", "").lower() == "q"


def run_human_mode() -> None:
    """Mode 1: Human plays Snake via keyboard."""
    if pygame is None:
        raise ImportError(
            "pygame is required for mode 1. Install with: pip install pygame"
        )

    env = SnakeEnv(fps=10)
    action = Action.STRAIGHT
    env.render(mode="human")

    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if _is_exit_key(event):
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RIGHT:
                        action = Action.TURN_RIGHT
                    elif event.key == pygame.K_LEFT:
                        action = Action.TURN_LEFT
                    elif event.key == pygame.K_UP:
                        action = Action.STRAIGHT

            _, _, done, info = env.step(int(action))
            env.render(mode="human")

            if done:
                print(f"Game Over. Score: {info['score']}")
                env.reset()

            action = Action.STRAIGHT
    finally:
        env.close()


def run_training_mode(agent, episodes: int = 500) -> None:
    """Mode 2: Simple training loop scaffold for RL integration."""
    env = SnakeEnv(fps=60)

    for episode in range(1, episodes + 1):
        state = env.reset()
        action = agent.get_action(state)
        done = False
        total_reward = 0.0
        i = 0
        while not done and i < 1000:
            next_state, reward, done, info = env.step(action)
            next_action = agent.get_action(next_state) if not done else None
            agent.update(state, action, next_state, next_action, reward, done)
            state = next_state
            action = next_action
            total_reward += reward
            i += 1
        print(
            f"Episode {episode:03d} | Score: {info['score']:02d} | "
            f"Steps: {info['steps']:03d} | Reward: {total_reward:.2f}"
        )
        agent.save_checkpoint("q_table.pkl")

    env.close()


def run_model_mode(
    model: Optional[Callable[[np.ndarray], int]] = None,
    episodes: int = 100,
) -> None:
    """Mode 3: Trained model/policy plays the game."""
    env = SnakeEnv(fps=12)
    policy = model.get_action if model is not None else RandomPolicy().predict

    try:
        rewards = []
        for ep in range(1, episodes + 1):
            state = env.reset()
            # env.render(mode="human")
            done = False
            while not done:
                # for event in pygame.event.get() if pygame is not None else []:
                #     if event.type == pygame.QUIT:
                #         return
                #     if _is_exit_key(event):
                #         return
                action = int(policy(state))
                state, _, done, info = env.step(action)
                # env.render(mode="human")
            rewards.append(info["score"])
            print(
                f"Model Episode {ep}: " f"score={info['score']}, steps={info['steps']}"
            )

        with open(os.path.join("result", "snake_q_table.txt"), "w") as file:
            file.write(str(sum(rewards) / len(rewards)))
    finally:
        env.close()


def main(agent=None) -> None:
    print("Choose mode:")
    print("1 - Human plays game")
    print("2 - Training mode")
    print("3 - Model plays game")

    try:
        mode = int(input("Input mode (1/2/3): ").strip())
    except ValueError:
        print("Invalid input.")
        return

    if mode == 1:
        run_human_mode()
    elif mode == 2:
        run_training_mode(agent, episodes=1000)
    elif mode == 3:
        agent = agent.load_checkpoint("q_table.pkl")
        run_model_mode(agent)
    else:
        print("Mode must be 1, 2, or 3.")


agent = QTableAgent(11, 3, 0.9, 0.5, 0.999, 1e-3, True)
main(agent)
