from env.snake.snake import *
from src.q_learning.dqn import DQAgent
import pygame
import random
import numpy as np
import os


class RandomPolicy:
    """Fallback policy for model mode demo."""

    def predict(self, state: np.ndarray, inference_mode=True) -> int:
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


def run_training_mode(agent, episodes: int = 500, batch_size=32) -> None:
    """Mode 2: Simple training loop scaffold for RL integration."""
    env = SnakeEnv(fps=60)
    best = -1e6
    for episode in range(1, episodes + 1):
        state = env.reset()
        action = agent.get_action(list(state))
        done = False
        total_reward = 0.0
        i = 0
        while not done and i < 1000:
            next_state, reward, done, info = env.step(action)
            agent.push(state, action, next_state, reward, done)
            agent.update_main(batch_size)
            action = agent.get_action(list(next_state))
            state = next_state
            total_reward += reward
            i += 1
        print(
            f"Episode {episode:03d} | Score: {info['score']:02d} | "
            f"Steps: {info['steps']:03d} | Reward: {total_reward:.2f} | Epsilon: {agent.current_epsilon:.4f}"
        )
        path = os.path.join("checkpoint", "dqn")
        if total_reward > best:
            best = total_reward
            agent.save_checkpoint(path, True)
        agent.save_checkpoint(path)
        agent.decay_epsilon(episode, episodes)
        if episode % 10 == 0:
            agent.update_target()

    env.close()


def run_model_mode(
    model: Optional[Callable[[np.ndarray], int]] = None,
    episodes: int = 100,
) -> None:
    """Mode 3: Trained model/policy plays the game."""
    # env = SnakeEnv(fps=12)
    env = SnakeEnv(fps=60)
    policy = model.get_action if model is not None else RandomPolicy().predict
    rewards = []
    try:

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
                action = int(policy(list(state), True))
                state, _, done, info = env.step(action)
                # env.render(mode="human")
            rewards.append(info["score"])
            print(
                f"Model Episode {ep}: " f"score={info['score']}, steps={info['steps']}"
            )
        with open(os.path.join("result", "snake_dqn.txt"), "w") as file:
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
        path = os.path.join("checkpoint", "dqn")
        agent.load_checkpoint(path, True)
        run_training_mode(agent, episodes=5000)
    elif mode == 3:
        path = os.path.join("checkpoint", "dqn")
        agent.load_checkpoint(path, True)
        run_model_mode(agent)
    else:
        print("Mode must be 1, 2, or 3.")


agent = DQAgent(11, 3, 1e-3, 1e-5, 0.5, 0.01, 1e-4, 0.9, 0.1)
main(agent)
