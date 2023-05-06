import gymnasium
from gymnasium import spaces
import numpy as np
import pygame
from typing import List, Optional, Tuple, Any

# from src.constants import WINDOW_SIZE
WINDOW_SIZE = 500


class Mpd(gymnasium.Env):
    """
    The Mdp is an implementation of the GridWorld environment described
    in Ng & Russell's paper "Algorithms for Inverse Reinforcement Learning".

    The agent moves in a 2D grid with the goal of reaching the target location.
    The agent can take four actions: up, down, left, and right.
    Each action can be set to have a small chance of moving in a random direction.

    Args:
        * size (int): The size of the square grid. Default is 5.
        * init_loc (tuple, optional): A tuple (y, x) indicating the starting location
            of the agent. If not provided, the agent starts in the lower left corner ()
        * targ_loc (tuple, optional): A tuple (y, x) indicating the target location.
            If not provided, the target is in the upper right corner.
        * targ_rwd (float, optional): The reward for reaching the target. Default is 1.
        * step_rwd (float, optional): The reward for each step. Default is 0.
        * noise (float, optional): The probability of the agent moving in a random
            direction instead of the intended direction. Default is 0.3.
        * render_mode (str, optional): The mode for rendering the environment.
            Options are "human" for Pygame-based rendering and "rgb_array"
            for returning the environment state as an RGB array. Default is None.

    Raises:
        ValueError: If the provided init_loc or targ_loc is not valid.

    Returns:
        None
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        size: int = 5,
        init_loc: Optional[Tuple[int, int]] = None,
        targ_loc: Optional[Tuple[int, int]] = None,
        targ_rwd: float = 1,
        step_rwd: float = 0,
        noise: float = 0.3,
        render_mode: Optional[str] = None,
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
            )
        else:
            self._init_loc = init_loc

        if targ_loc is None:
            self._targ_loc = (
                0,
                self.size - 1,
            )
        else:
            self._targ_loc = targ_loc

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(4)

        # Maps action values in integers to changes in coordinates with dict.
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

    def _get_obs(self) -> np.ndarray:
        """
        Returns the agent's current location as a 1D numpy array of shape (2,).

        Returns:
            np.ndarray: The agent's current location in the grid as a 1D numpy array [row, column].
        """
        return self._agent_location

    def _state_to_ind(self, state) -> int:
        """
        Converts a 2D grid state (numpy array) to a 1D index.

        Args:
            state (np.ndarray): A 1D numpy array of shape (2,) representing the row and column coordinates of the state.

        Returns:
            int: The corresponding 1D index of the input state.
        """
        return int(state[0] * self.size + state[1])

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Resets the environment to its initial state.
        Sets the agent's initial location based on _init_loc,
        and sets the target location. Returns the initial observation.

        Args:
            seed (Optional[int], optional): The random seed for reproducibility. Defaults to None.

        Raises:
            ValueError: If the provided _init_loc value is of an invalid shape.

        Returns:
            np.ndarray: The initial observation of the agent's location as a 1D numpy array [row, column].
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

    def _get_neighbours(self, state_pos: np.ndarray) -> List[int]:
        """
        Finds the neighboring states of the given state in the grid.

        Args:
            state_pos (np.ndarray): A 1D numpy array of shape (2,) representing the
            row and column coordinates of the state.

        Returns:
            List[int]: A list of 1D indices corresponding to the neighboring states.
        """
        neighbours = set()
        for a in range(len(self._action_to_direction)):
            direction = self._action_to_direction[a]
            neighbour = np.clip(state_pos + direction, 0, self.size - 1)
            neighbours.add(self._state_to_ind(neighbour))
        neighbours = list(neighbours)
        return neighbours

    def get_p(self) -> np.ndarray:
        """
        Calculates the probability distribution for all state transitions,
        given a state and an action. Returns a 4D numpy array representing
        the transition probabilities.

        Returns:
            np.ndarray: A 4D numpy array of shape (size^2, 2, size^2, 2, len(_action_to_direction))
            representing the transition probabilities.
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

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Any]:
        """
        Executes the given action in the environment, updating the agent's location.
        Returns the resulting observation, reward, termination flag, and additional information.

        Args:
            action (int): The action to be executed in the environment.

        Returns:
            tuple: A tuple containing the observation, reward, termination flag, and additional information.
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

    def render(self) -> Optional[np.ndarray]:
        """
        Renders the environment in a human-readable format using Pygame or as an RGB array.

        Returns:
            np.ndarray: The RGB array representation of the environment if the render mode is "rgb_array".
        """
        if self.render_mode == "human":
            self._render_frame()
        elif self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        """
        Helper method for rendering the environment's current state.

        Returns:
            np.ndarray: The RGB array representation of the environment if the render mode is "rgb_array".
        """
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((128, 128, 128))

        # The size of a single grid square in pixels
        pix_square_size = self.window_size / self.size

        # Load target image
        treasure_img = pygame.image.load("cheese.png")
        treasure_img = pygame.transform.scale(
            treasure_img, (int(pix_square_size), int(pix_square_size))
        )

        # Set coordinates for target
        treasure_pos = (
            (self._targ_loc[1] * pix_square_size),
            (self._targ_loc[0] * pix_square_size),
        )

        # Draw target
        canvas.blit(treasure_img, treasure_pos)

        # Load image of Morris
        morris_img = pygame.image.load("morris.png")
        morris_img = pygame.transform.scale(
            morris_img, (int(pix_square_size), int(pix_square_size))
        )

        # Coordinates of Morris
        morris_pos = (
            (self._agent_location[1] * pix_square_size),
            (self._agent_location[0] * pix_square_size),
        )

        # Draw morris
        canvas.blit(morris_img, morris_pos)

        # Finally, add some gridlines
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
            # The following line copies our drawings from `canvas` to the visible window
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

    def close(self) -> None:
        """Closes the rendering window if it is open."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
