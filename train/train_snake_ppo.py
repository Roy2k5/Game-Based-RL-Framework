from env.snake.snake import *
from src.ppo.ppo import ppo
from src.ppo.core import MLPCategoricalActor
import pygame
import random
import numpy as np
import os
from torch import nn
import torch


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


def run_training_mode(ckpt_dir) -> None:
    """Mode 2: Simple training loop scaffold for RL integration."""
    env = SnakeEnv(fps=60)
    ppo(lambda: env, ckpt_dir=ckpt_dir)
    env.close()


def run_model_mode(
    model: Optional[Callable[[np.ndarray], int]] = None,
    episodes: int = 100,
) -> None:
    """Mode 3: Trained model/policy plays the game."""
    env = SnakeEnv(fps=12)
    # env = SnakeEnv(fps=60)
    policy = model.get_action if model is not None else RandomPolicy().predict
    rewards = []
    try:

        for ep in range(1, episodes + 1):
            state = env.reset()
            env.render(mode="human")
            done = False
            while not done:
                for event in pygame.event.get() if pygame is not None else []:
                    if event.type == pygame.QUIT:
                        return
                    if _is_exit_key(event):
                        return
                    action = int(policy(list(state)))
                    state, _, done, info = env.step(action)
                    env.render(mode="human")
            rewards.append(info["score"])
            print(
                f"Model Episode {ep}: " f"score={info['score']}, steps={info['steps']}"
            )
        with open(os.path.join("result", "snake_ppo.txt"), "w") as file:
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
    ckpt_dir = os.path.join("checkpoint", "ppo")

    if mode == 1:
        run_human_mode()
    elif mode == 2:
        run_training_mode(ckpt_dir)
    elif mode == 3:
        filepath = os.path.join(ckpt_dir, "ppo_pi_last.pt")
        agent.load_state_dict(
            torch.load(filepath, map_location="cpu", weights_only=True)
        )
        run_model_mode(agent)
    else:
        print("Mode must be 1, 2, or 3.")


agent = MLPCategoricalActor(11, 3, (32, 32), nn.Tanh)

main(agent)
