from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

try:
    import pygame
except ImportError:
    pygame = None


State = Tuple[int, int, int, int]


@dataclass
class CustomReward:
    """Reward shaping helper for Flappy Bird RL."""

    survive: float = 0.05
    pass_pipe: float = 10.0
    crash: float = -10.0
    center_bonus: float = 0.1

    def compute(
        self, crashed: bool, passed_pipe: bool, center_distance: float
    ) -> float:
        if crashed:
            return self.crash
        if passed_pipe:
            return self.pass_pipe
        return self.survive + self.center_bonus * max(0.0, 1.0 - center_distance)


class FlappyBirdEnv:
    """Simple Flappy Bird environment for RL.

    Action space:
        0: do nothing
        1: flap

    Observation (discretized tuple, 4 dims):
        (dx_to_pipe, dy_to_gap_center, velocity_bin, is_pipe_ahead)
    """

    def __init__(
        self,
        width: int = 420,
        height: int = 640,
        bird_x: int = 100,
        bird_size: int = 24,
        gravity: float = 0.55,
        flap_velocity: float = -8.5,
        pipe_width: int = 70,
        pipe_speed: float = 3.0,
        pipe_gap: int = 170,
        min_gap_y: int = 120,
        max_gap_y: int = 520,
        fps: int = 60,
        reward_fn: Optional[CustomReward] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.width = width
        self.height = height
        self.bird_x = bird_x
        self.bird_size = bird_size
        self.gravity = gravity
        self.flap_velocity = flap_velocity
        self.pipe_width = pipe_width
        self.pipe_speed = pipe_speed
        self.pipe_gap = pipe_gap
        self.min_gap_y = min_gap_y
        self.max_gap_y = max_gap_y
        self.fps = fps
        self.reward_fn = reward_fn or CustomReward()

        self._rng = random.Random(seed)

        self.bird_y: float = 0.0
        self.bird_velocity: float = 0.0
        self.pipe_x: float = 0.0
        self.gap_center_y: float = 0.0
        self.score: int = 0
        self.steps: int = 0
        self.done: bool = False
        self._pipe_counted: bool = False

        self._pygame_ready = False
        self._screen = None
        self._clock = None

        self.reset()

    def reset(self) -> State:
        self.bird_y = float(self.height // 2)
        self.bird_velocity = 0.0
        self.pipe_x = float(self.width + 60)
        self.gap_center_y = float(self._rng.randint(self.min_gap_y, self.max_gap_y))
        self.score = 0
        self.steps = 0
        self.done = False
        self._pipe_counted = False
        return self._get_state()

    def step(self, action: int) -> Tuple[State, float, bool, Dict[str, int]]:
        if self.done:
            return (
                self._get_state(),
                0.0,
                True,
                {"score": self.score, "steps": self.steps},
            )

        action = int(action)
        if action not in (0, 1):
            raise ValueError("Action must be one of [0, 1].")

        if action == 1:
            self.bird_velocity = self.flap_velocity

        self.bird_velocity += self.gravity
        self.bird_y += self.bird_velocity
        self.pipe_x -= self.pipe_speed
        self.steps += 1

        passed_pipe = False
        bird_front_x = self.bird_x + self.bird_size / 2
        pipe_back_x = self.pipe_x + self.pipe_width
        if not self._pipe_counted and pipe_back_x < bird_front_x:
            self.score += 1
            passed_pipe = True
            self._pipe_counted = True

        if self.pipe_x + self.pipe_width < 0:
            self.pipe_x = float(self.width)
            self.gap_center_y = float(self._rng.randint(self.min_gap_y, self.max_gap_y))
            self._pipe_counted = False

        crashed = self._is_collision()
        self.done = crashed

        center_distance = abs(self.bird_y - self.gap_center_y) / max(
            1.0, self.pipe_gap / 2
        )
        reward = self.reward_fn.compute(crashed, passed_pipe, center_distance)
        info = {"score": self.score, "steps": self.steps}

        return self._get_state(), float(reward), self.done, info

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        if mode not in ("human", "rgb_array"):
            raise ValueError("mode must be 'human' or 'rgb_array'.")

        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame[:, :] = (140, 200, 255)

        gap_top = int(self.gap_center_y - self.pipe_gap / 2)
        gap_bottom = int(self.gap_center_y + self.pipe_gap / 2)
        px0 = max(0, int(self.pipe_x))
        px1 = min(self.width, int(self.pipe_x + self.pipe_width))

        if px0 < px1:
            frame[0:gap_top, px0:px1] = (0, 180, 0)
            frame[gap_bottom : self.height, px0:px1] = (0, 180, 0)

        by0 = max(0, int(self.bird_y - self.bird_size / 2))
        by1 = min(self.height, int(self.bird_y + self.bird_size / 2))
        bx0 = max(0, int(self.bird_x - self.bird_size / 2))
        bx1 = min(self.width, int(self.bird_x + self.bird_size / 2))
        frame[by0:by1, bx0:bx1] = (255, 220, 0)

        if mode == "rgb_array":
            return frame

        if pygame is None:
            raise ImportError(
                "pygame is required for human rendering. "
                "Install with: pip install pygame"
            )

        self._init_pygame_if_needed()
        surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        self._screen.blit(surface, (0, 0))
        self._draw_score()
        pygame.display.flip()
        self._clock.tick(self.fps)
        return None

    def close(self) -> None:
        if self._pygame_ready and pygame is not None:
            pygame.quit()
        self._pygame_ready = False
        self._screen = None
        self._clock = None

    def _is_collision(self) -> bool:
        bird_top = self.bird_y - self.bird_size / 2
        bird_bottom = self.bird_y + self.bird_size / 2

        if bird_top < 0 or bird_bottom >= self.height:
            return True

        bird_left = self.bird_x - self.bird_size / 2
        bird_right = self.bird_x + self.bird_size / 2

        pipe_left = self.pipe_x
        pipe_right = self.pipe_x + self.pipe_width

        horizontal_hit = bird_right >= pipe_left and bird_left <= pipe_right
        if not horizontal_hit:
            return False

        gap_top = self.gap_center_y - self.pipe_gap / 2
        gap_bottom = self.gap_center_y + self.pipe_gap / 2
        in_gap = bird_top >= gap_top and bird_bottom <= gap_bottom
        return not in_gap

    def _get_state(self) -> State:
        dx = int(round((self.pipe_x - self.bird_x) / 20.0))
        dy = int(round((self.bird_y - self.gap_center_y) / 15.0))
        vel = int(round(self.bird_velocity * 2.0))

        dx = int(np.clip(dx, -10, 20))
        dy = int(np.clip(dy, -15, 15))
        vel = int(np.clip(vel, -20, 20))

        is_pipe_ahead = int(self.pipe_x + self.pipe_width >= self.bird_x)
        return (dx, dy, vel, is_pipe_ahead)

    def _init_pygame_if_needed(self) -> None:
        if self._pygame_ready:
            return
        pygame.init()
        self._screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Flappy Bird RL Environment")
        self._clock = pygame.time.Clock()
        self._pygame_ready = True

    def _draw_score(self) -> None:
        font = pygame.font.SysFont("arial", 26)
        text = font.render(f"Score: {self.score}", True, (40, 40, 40))
        self._screen.blit(text, (12, 12))


class RandomPolicy:
    """Fallback policy for model mode demo."""

    def predict(self, state: State) -> int:
        _ = state
        return 1 if random.random() < 0.12 else 0


def _is_exit_key(event: Any) -> bool:
    if event.type != pygame.KEYDOWN:
        return False

    if event.key in (pygame.K_q, pygame.K_ESCAPE):
        return True

    return getattr(event, "unicode", "").lower() == "q"


def run_human_mode() -> None:
    """Mode 1: Human plays Flappy Bird."""
    if pygame is None:
        raise ImportError(
            "pygame is required for mode 1. Install with: pip install pygame"
        )

    env = FlappyBirdEnv(fps=60)
    env.render(mode="human")

    try:
        while True:
            action = 0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if _is_exit_key(event):
                    return
                if event.type == pygame.KEYDOWN and event.key in (
                    pygame.K_SPACE,
                    pygame.K_UP,
                ):
                    action = 1

            _, _, done, info = env.step(action)
            env.render(mode="human")

            if done:
                print(f"Game Over. Score: {info['score']}")
                env.reset()
    finally:
        env.close()


def run_training_mode(agent, episodes: int = 500) -> None:
    """Mode 2: Training loop for Q-learning/SARSA integration."""
    if agent is None:
        raise ValueError("agent is required for training mode")

    env = FlappyBirdEnv(fps=120)

    for episode in range(1, episodes + 1):
        state = env.reset()
        action = agent.get_action(state)
        done = False
        total_reward = 0.0

        while not done:
            next_state, reward, done, info = env.step(action)
            next_action = agent.get_action(next_state) if not done else None
            agent.update(state, action, next_state, next_action, reward, done)
            state = next_state
            action = next_action if next_action is not None else 0
            total_reward += reward

        print(
            f"Episode {episode:03d} | Score: {info['score']:02d} | "
            f"Steps: {info['steps']:04d} | Reward: {total_reward:.2f}"
        )
        agent.save_checkpoint("flappy_q_table.pkl")

    env.close()


def run_model_mode(
    model: Optional[Callable[[State], int]] = None,
    episodes: int = 50,
) -> None:
    """Mode 3: Trained model/policy plays Flappy Bird."""
    env = FlappyBirdEnv(fps=60)
    policy = model.get_action if model is not None else RandomPolicy().predict

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

                action = int(policy(state))
                state, _, done, info = env.step(action)
                env.render(mode="human")

            print(f"Model Episode {ep}: score={info['score']}, steps={info['steps']}")
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
        if agent is None:
            run_model_mode(None)
        else:
            agent = agent.load_checkpoint("flappy_q_table.pkl")
            run_model_mode(agent)
    else:
        print("Mode must be 1, 2, or 3.")


if __name__ == "__main__":
    main()
