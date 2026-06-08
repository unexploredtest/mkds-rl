import random
import enum

import pynds
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# One lap is about 1360 distance
# A decent run finishes one lap in about 2300 frames

# KEY_MAP = {
#     'a': 0,
#     'b': 1,
#     'select': 2,
#     'start': 3,
#     'right': 4,
#     'left': 5,
#     'up': 6,
#     'down': 7,
#     'r': 8,
#     'l': 9,
#     'x': 10,
#     'y': 11
# }

RAM_ADDRESSES = {
    "back_distance": 3513708,
    "front_distance": 3497980
}

class Actions(enum.Enum):
    NONE = 0
    LEFT = 1
    RIGHT = 2

class MarioKartDSEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    lap_distance = 1360
    distance_timeout = 60

    def __init__(self, rom_path: str, savestates: list[str] = [], render_mode=None):
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(192, 256, 4), dtype=np.uint8)
        self.action_space = spaces.Discrete(3)

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        i.e. 0 corresponds to "right", 1 to "up" etc.
        Uses NumPy [row, col] convention where row 0 is at the top.
        """

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window_opened = False
        self.clock = None

        if(isinstance(savestates, str)):
            savestates = [savestates]
        assert isinstance(savestates, list)
        self.savestates = savestates

        self.nds = pynds.PyNDS(rom_path)

        self.current_action = Actions.NONE

        # Distance related
        self.distance = None
        self.max_distance = None
        self.last_max_distance = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # load savestate
        self.nds.load_state_from_file(random.choice(self.savestates))

        # progress
        self.nds.tick()
        
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            if(not self.window_opened):
                self.nds.open_window()
                self.window_opened = True

            self.nds.render()

        self.current_action = Actions.NONE
        self.distance = self.nds.memory.read_ram_i32(RAM_ADDRESSES["back_distance"])
        self.max_distance = self.distance
        self.last_max_distance = 0

        return observation, info

    def step(self, action):        
        new_action = Actions(action)

        self.nds.button.press_key('a')
        if(new_action == Actions.NONE):
            self.nds.button.release_key('left')
            self.nds.button.release_key('right')
        elif(new_action == Actions.LEFT):
            self.nds.button.press_key('left')
            self.nds.button.release_key('right')
        elif(new_action == Actions.RIGHT):
            self.nds.button.release_key('left')
            self.nds.button.press_key('right')
        else:
            raise ValueError(f"{action} is not a valid action!")

        self.nds.tick()

        new_distance = self.nds.memory.read_ram_i32(RAM_ADDRESSES["back_distance"])

        # Checking whether the agent has progressed through the truck or stuck
        if(new_distance <= self.max_distance):
            self.last_max_distance += 1
        else:
            self.max_distance = new_distance
            self.last_max_distance = 0

        # We terminate when the agent finishes the lap or when the agent hasn't progressed through the track
        terminated = new_distance > self.lap_distance + 5 or self.last_max_distance > self.distance_timeout
        reward = self._get_reward(new_distance)
        observation = self._get_obs()
        info = self._get_info()

        self.distance = new_distance

        if self.render_mode == "human":
            self.nds.render()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def close(self):
        if(self.window_opened):
            self.nds.close_window()

    def _get_obs(self):
        top_frame, bottom_frame = self.nds.get_frame()
        return top_frame

    def _get_info(self):
        return {
            "distance": self.nds.memory.read_ram_i32(RAM_ADDRESSES["back_distance"])
        }

    def _get_reward(self, new_distance: int):
        reward_clipped = np.clip(new_distance - self.distance, -10, 10)
        reward_normalized = reward_clipped / 100
        return float(reward_normalized)

    def _render_frame(self):
        top_frame, bottom_frame = self.nds.get_frame()
        merged = np.vstack((top_frame, bottom_frame))
        return merged

gym.envs.register(
    id="MarioKartDS-v0",
    entry_point=MarioKartDSEnv,
    max_episode_steps=3000
)

if __name__ == "__main__":
    # env = MarioKartDSEnv("files/rom.nds", "savestates/time_trail_begining.noo", render_mode="human")
    env = gym.make("MarioKartDS-v0", rom_path="files/rom.nds", savestates="savestates/time_trail_begining.noo", render_mode="human")
    obs, info = env.reset()

    done = False
    i = 0
    while(not done):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        i += 1
        done = terminated | truncated
        # print(i)
        # print(reward)