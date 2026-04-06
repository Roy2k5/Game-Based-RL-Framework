from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Callable, Deque, Dict, Optional, Tuple

import numpy as np

try:
    import pygame
except ImportError:
    pygame = None


Position = Tuple[int, int]


class Action(IntEnum):
    STRAIGHT = 0
    TURN_RIGHT = 1
    TURN_LEFT = 2


class Direction(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


@dataclass
class CustomReward:
    """Reward shaping helper for Snake RL."""

    eat_food: float = 10.0
    die: float = -10.0
    step: float = -0.05
    toward_food: float = 0.3
    away_from_food: float = -0.3

    def compute(
        self,
        ate_food: bool,
        dead: bool,
        old_head: Position,
        new_head: Position,
        food: Position,
    ) -> float:
        if dead:
            return self.die
        if ate_food:
            return self.eat_food

        old_dist = abs(old_head[0] - food[0]) + abs(old_head[1] - food[1])
        new_dist = abs(new_head[0] - food[0]) + abs(new_head[1] - food[1])

        shaping = self.toward_food if new_dist < old_dist else self.away_from_food
        return self.step + shaping


class SnakeEnv:
    """Simple Snake environment for RL training and evaluation.

    Action space:
        0: keep moving straight
        1: turn right
        2: turn left

    Observation (11-dim):
        [danger_straight, danger_right, danger_left,
         dir_up, dir_right, dir_down, dir_left,
         food_up, food_right, food_down, food_left]
    """

    def __init__(
        self,
        width: int = 60,
        height: int = 35,
        block_size: int = 20,
        fps: int = 12,
        max_steps_without_food: int = 300,
        reward_fn: Optional[CustomReward] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.width = width
        self.height = height
        self.block_size = block_size
        self.fps = fps
        self.max_steps_without_food = max_steps_without_food
        self.reward_fn = reward_fn or CustomReward()
        self.obs_dim = 11
        self.action_dim = 3
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

        self.direction: Direction = Direction.RIGHT
        self.snake: Deque[Position] = deque()
        self.food: Position = (0, 0)
        self.score: int = 0
        self.steps: int = 0
        self.steps_since_food: int = 0
        self.done: bool = False

        self._pygame_ready = False
        self._screen = None
        self._clock = None

        self.reset()

    def reset(self) -> np.ndarray:
        cx = self.width // 2
        cy = self.height // 2
        self.direction = Direction.RIGHT
        self.snake = deque([(cx, cy), (cx - 1, cy), (cx - 2, cy)])
        self.food = self._spawn_food()
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0
        self.done = False
        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, int]]:
        if self.done:
            return self._get_state(), 0.0, True, {"score": self.score}

        action = int(action)
        if action not in (0, 1, 2):
            raise ValueError("Action must be one of [0, 1, 2].")

        old_head = self.snake[0]
        self._update_direction(Action(action))
        new_head = self._next_head(old_head, self.direction)

        self.steps += 1
        self.steps_since_food += 1

        dead = self._is_collision(new_head)
        ate_food = False
        old_food = self.food
        if not dead:
            self.snake.appendleft(new_head)
            if new_head == self.food:
                ate_food = True
                self.score += 1
                self.steps_since_food = 0
                self.food = self._spawn_food()
            else:
                self.snake.pop()

        if self.steps_since_food >= self.max_steps_without_food:
            dead = True

        self.done = dead
        reward = self.reward_fn.compute(ate_food, dead, old_head, new_head, old_food)
        info = {"score": self.score, "steps": self.steps}

        return self._get_state(), float(reward), self.done, info

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render the environment.

        mode='human': display with pygame.
        mode='rgb_array': return a frame as np.ndarray(H, W, 3).
        """
        if mode not in ("human", "rgb_array"):
            raise ValueError("mode must be 'human' or 'rgb_array'.")

        frame = np.zeros(
            (self.height * self.block_size, self.width * self.block_size, 3),
            dtype=np.uint8,
        )

        frame[:, :] = (20, 20, 20)

        for x, y in self.snake:
            x1 = x * self.block_size
            y1 = y * self.block_size
            frame[y1 : y1 + self.block_size, x1 : x1 + self.block_size] = (
                0,
                200,
                0,
            )

        fx, fy = self.food
        x1 = fx * self.block_size
        y1 = fy * self.block_size
        frame[y1 : y1 + self.block_size, x1 : x1 + self.block_size] = (220, 40, 40)

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

    def _spawn_food(self) -> Position:
        all_cells = [(x, y) for x in range(self.width) for y in range(self.height)]
        snake_set = set(self.snake)
        free_cells = [p for p in all_cells if p not in snake_set]
        if not free_cells:
            return self.snake[0]
        return self._rng.choice(free_cells)

    def _update_direction(self, action: Action) -> None:
        current = self.direction
        if action == Action.STRAIGHT:
            return
        if action == Action.TURN_RIGHT:
            self.direction = Direction((current + 1) % 4)
        elif action == Action.TURN_LEFT:
            self.direction = Direction((current - 1) % 4)

    def _next_head(self, head: Position, direction: Direction) -> Position:
        x, y = head
        if direction == Direction.UP:
            return x, y - 1
        if direction == Direction.RIGHT:
            return x + 1, y
        if direction == Direction.DOWN:
            return x, y + 1
        return x - 1, y

    def _is_collision(self, point: Position) -> bool:
        x, y = point
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True
        if point in self.snake:
            return True
        return False

    def _get_state(self) -> np.ndarray:
        head = self.snake[0]
        dir_up = self.direction == Direction.UP
        dir_right = self.direction == Direction.RIGHT
        dir_down = self.direction == Direction.DOWN
        dir_left = self.direction == Direction.LEFT

        straight_dir = self.direction
        right_dir = Direction((self.direction + 1) % 4)
        left_dir = Direction((self.direction - 1) % 4)

        point_straight = self._next_head(head, straight_dir)
        point_right = self._next_head(head, right_dir)
        point_left = self._next_head(head, left_dir)

        fx, fy = self.food
        hx, hy = head

        state = (
            int(self._is_collision(point_straight)),
            int(self._is_collision(point_right)),
            int(self._is_collision(point_left)),
            int(dir_up),
            int(dir_right),
            int(dir_down),
            int(dir_left),
            int(fy < hy),
            int(fx > hx),
            int(fy > hy),
            int(fx < hx),
        )
        return state

    def _init_pygame_if_needed(self) -> None:
        if self._pygame_ready:
            return
        pygame.init()
        self._screen = pygame.display.set_mode(
            (self.width * self.block_size, self.height * self.block_size)
        )
        pygame.display.set_caption("Snake RL Environment")
        self._clock = pygame.time.Clock()
        self._pygame_ready = True

    def _draw_score(self) -> None:
        font = pygame.font.SysFont("arial", 20)
        text = font.render(f"Score: {self.score}", True, (240, 240, 240))
        self._screen.blit(text, (8, 6))
