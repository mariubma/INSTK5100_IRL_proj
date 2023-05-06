import gymnasium
from gymnasium import spaces
import numpy as np
import pygame
import os

# from src.constants import WINDOW_SIZE
WINDOW_SIZE = 500


class GridWorld(gymnasium.Env):
    """
    GridWorld is an implementation of the GridWorld environment described
    in Ng & Russell's paper "Algorithms for Inverse Reinforcement Learning".

    The agent moves in a 2D grid with the goal of reaching the target location.
    The agent can take four actions: up, down, left, and right.
    Each action can be set to have a small chance of moving in a random direction.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        size=5,  # Ng & Russell's gridworld is 5x5
        init_loc=None,
        targ_loc=None,
        targ_rwd=1,  # Ng & Russell's target reward is 1
        step_rwd=0,
        noise=0.3,
        render_mode=None,
    ):
        self.size = size
        self.window_size = WINDOW_SIZE
        self._target_reward = targ_rwd
        self._step_reward = step_rwd
        self._noise = noise

        if init_loc is None:
            self._init_loc = (
                self.size - 1,
                0,
            )  # Ng & Russell's agent starts in the lower left corner
        else:
            self._init_loc = init_loc

        if targ_loc is None:
            self._targ_loc = (
                0,
                self.size - 1,
            )  # Ng & Russell's target is the upper right corner
        else:
            self._targ_loc = targ_loc

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(4)
        self._action_to_direction = {
            0: np.array([1, 0]),  # Down
            1: np.array([0, 1]),  # Right
            2: np.array([-1, 0]),  # Up
            3: np.array([0, -1]),  # Left
        }

        self._ind_to_state = [
            np.array(
                [
                    (state_ind - (state_ind % self.size)) / self.size,
                    state_ind % self.size,
                ]
            )
            for state_ind in range(self.size**2)
        ]
        """
        With human-rendering, `self.window` references the window that we draw to.
        `self.clock` is used to ensure the environment renders at the right framerate. 
        Both remain `None` until human-mode is used for the first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        """Returns the agent's current location."""
        return self._agent_location

    def _state_to_ind(self, state):
        """Converts a 2D grid state to a 1D index."""
        return int(state[0] * self.size + state[1])

    def reset(self, seed=None):
        """
        Resets the environment to its initial state.
        Sets the agent's initial location based on _init_loc,
        and sets the target location. Returns the initial observation.
        """
        super().reset(seed=seed)
        if self._init_loc is not None:
            if np.shape(self._init_loc) == (2,):
                self._agent_location = self._init_loc
            elif np.shape(self._init_loc) == ():
                self._agent_location = self._ind_to_state[self._init_loc]
            else:
                raise ValueError("Invalid init_loc")
        else:
            self._agent_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        self._target_location = self._targ_loc

        observation = self._get_obs()

        if self.render_mode == "human":
            self._render_frame()

        return observation

    def _get_neighbours(self, state_pos):
        neighbours = set()
        for a in range(len(self._action_to_direction)):
            direction = self._action_to_direction[a]
            neighbour = np.clip(state_pos + direction, 0, self.size - 1)
            neighbours.add(self._state_to_ind(neighbour))
        neighbours = list(neighbours)
        return neighbours

    def get_p(self):
        """
        Calculates the probability distribution for all state transitions,
        given a state and an action. Returns a 4D numpy array representing
        the transition probabilities.
        """
        p = np.zeros(
            (self.size**2, 2, self.size**2, len(self._action_to_direction))
        )
        for state_ind in range(self.size**2):
            state_pos = self._ind_to_state[state_ind]
            neighbours = self._get_neighbours(state_pos)
            for next_state_ind in neighbours:
                for action in range(4):
                    direction = self._action_to_direction[action]
                    desired_state = np.clip(state_pos + direction, 0, self.size - 1)
                    desired_state = self._state_to_ind(desired_state)

                    # If step is to target, then reward is 1, else 0
                    reward_ind = 0
                    if np.array_equal(
                        next_state_ind, self._state_to_ind(self._targ_loc)
                    ):
                        reward_ind = 1
                    if np.array_equal(next_state_ind, desired_state):
                        # Setting prob of taking step in intended direction
                        p[next_state_ind, reward_ind, state_ind, action] = (
                            1 - self._noise
                        ) + self._noise / 4
                    else:
                        # Setting prob of taking step in unintended direction
                        p[next_state_ind, reward_ind, state_ind, action] = (
                            self._noise / 4
                        )
        return p

    def step(self, action: int):
        """
        Takes an action in the environment, updating the agent's location
        and returning the observation, reward, termination flag, and other
        information.
        """
        terminated = False
        reward = self._step_reward
        direction = self._action_to_direction[action]
        u = np.random.uniform(0, 1)
        if u < self._noise:
            direction = self._action_to_direction[
                np.random.randint(0, len(self._action_to_direction) - 1)
            ]
        new_location = self._agent_location + direction
        is_invalid = (
            (new_location[0] < 0)
            or (new_location[1] < 0)
            or (new_location[0] > self.size - 1)
            or (new_location[1] > self.size - 1)
        )

        if is_invalid:
            new_location = self._agent_location
        else:
            self._agent_location = new_location
            if np.array_equal(self._agent_location, self._targ_loc):
                reward = self._target_reward
                terminated = True

        observation = self._get_obs()

        return observation, reward, terminated, False, None

    def render(self):
        """
        Renders the environment, either in human-readable format
        using Pygame or as an RGB array.
        """
        if self.render_mode == "human":
            self._render_frame()
        elif self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        """
        Helper method for rendering the environment's current state.
        """
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = self.window_size / self.size  # Size of a single square in pix

        # Target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._targ_loc,
                (pix_square_size, pix_square_size),
            ),
        )
        # Agent
        # pygame.draw.circle(
        #     canvas,
        #     (0, 0, 255),
        #     (self._agent_location + 0.5) * pix_square_size,
        #     pix_square_size / 3,
        # )

        self.agent_image = pygame.image.load("morris.png")
        scaled_agent_image = pygame.transform.scale(
            self.agent_image, (int(pix_square_size), int(pix_square_size))
        )
        agent_position = (self._agent_location * pix_square_size).astype(int)
        canvas.blit(scaled_agent_image, agent_position)

        # Gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        """Closes the rendering window if it is open."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
