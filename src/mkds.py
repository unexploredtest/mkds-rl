import enum

import gymnasium as gym
import numpy as np
import pynds
from gymnasium import spaces

# Memory addresses for important values in the game (used for rewards and extra info)
RAM_ADDRESSES_CHAIN = {
    "back_distance": (1552060, 0),
    "front_distance": (3497980,),
    "speed": (1552060, 1020),
    "total_time_elapsed": (1529372, 0),
    "lap_time_elapsed": (1529372, 24),
    "lap": (1554556,),
}

# An enum of all possible actions
class Actions(enum.Enum):
    NONE = 0
    LEFT = 1
    RIGHT = 2
    ITEM = 3
    DRIFT = 4
    DRIFT_LEFT = 5
    DRIFT_RIGHT = 6


class MarioKartDSEnv(gym.Env):
    metadata = {  # noqa: RUF012
        "render_modes": ["human", "rgb_array"],
        "render_fps": 60,
    }
    lap_distance = 1360
    distance_timeout = 60

    def __init__(
        self,
        rom_path: str,
        savestates: list[str] | str,
        rewards_weight: tuple[int] = (0, 0, 1),
        terminate_on_stall=True,
        render_mode=None,
    ):
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(192, 256, 4), dtype=np.uint8
        )
        self.action_space = spaces.Discrete(7) # Discrete actions that correspond to the enum Actions

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window_opened = False

        # Be sure that savestates is a list because a method is used in reset for random selection
        # that only operates on lists
        if isinstance(savestates, str):
            savestates = [savestates]

        assert isinstance(savestates, list)
        self.savestates = savestates

        # Create a PyNDS instance
        self.nds = pynds.PyNDS(rom_path)

        # Environment related values
        self.current_action = Actions.NONE
        self.checkpoint_size = 100
        self.rewards_weight = rewards_weight
        self.terminate_on_stall = terminate_on_stall

        # Game related values
        self.last_checkpoint = 0
        self.steps_in_checkpoint = 0
        self.win = 0
        self.current_lap = 1
        self.total_time = 0
        self.lap_elapsed = 0
        self.distance = None
        self.max_distance = None
        self.last_max_distance = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # load savestate
        self.nds.load_state_from_file(self.np_random.choice(self.savestates))

        # progress
        self.nds.tick()

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            if not self.window_opened:
                self.nds.open_window()
                self.window_opened = True

            self.nds.render()

        # Set initial values
        self.current_action = Actions.NONE
        self.steps_in_checkpoint = 0
        self.win = 0
        self.current_lap = self.nds.memory.read_ram_u32(
            self._read_mem_chain(RAM_ADDRESSES_CHAIN["lap"])
        )
        self.lap_elapsed = self.nds.memory.read_ram_u32(
            self._read_mem_chain(RAM_ADDRESSES_CHAIN["lap_time_elapsed"])
        )
        self.distance = self.nds.memory.read_ram_i32(
            self._read_mem_chain(RAM_ADDRESSES_CHAIN["back_distance"])
        )
        self.max_distance = self.distance
        self.last_checkpoint = self.distance
        self.last_max_distance = 0

        return observation, info

    def step(self, action):
        # Apply action
        new_action = Actions(action)

        self.nds.button.press_key("a") # We always move forward
        if new_action == Actions.NONE: # Nothing
            self.nds.button.release_key("x")
            self.nds.button.release_key("left")
            self.nds.button.release_key("right")
            self.nds.button.release_key("r")
        elif new_action == Actions.LEFT: # Turn left
            self.nds.button.release_key("x")
            self.nds.button.release_key("right")
            self.nds.button.release_key("r")
            self.nds.button.press_key("left")
        elif new_action == Actions.RIGHT: # Turn right
            self.nds.button.release_key("x")
            self.nds.button.release_key("left")
            self.nds.button.release_key("r")
            self.nds.button.press_key("right")
        elif new_action == Actions.ITEM: # Use item
            self.nds.button.release_key("left")
            self.nds.button.release_key("right")
            self.nds.button.release_key("r")
            self.nds.button.press_key("x")
        elif new_action == Actions.DRIFT: # Drift
            self.nds.button.release_key("left")
            self.nds.button.release_key("right")
            self.nds.button.release_key("x")
            self.nds.button.press_key("r")
        elif new_action == Actions.DRIFT_LEFT: # Drift left
            self.nds.button.release_key("right")
            self.nds.button.release_key("x")
            self.nds.button.press_key("left")
            self.nds.button.press_key("r")
        elif new_action == Actions.DRIFT_RIGHT: # Drift right
            self.nds.button.release_key("left")
            self.nds.button.release_key("x")
            self.nds.button.press_key("right")
            self.nds.button.press_key("r")
        else: # Invalid action
            raise ValueError(f"{action} is not a valid action!")

        self.nds.tick()

        new_distance = self.nds.memory.read_ram_i32(
            self._read_mem_chain(RAM_ADDRESSES_CHAIN["back_distance"])
        )

        # Checking whether the agent has progressed through the truck or stuck
        if new_distance <= self.max_distance:
            self.last_max_distance += 1
        else:
            self.max_distance = new_distance
            self.last_max_distance = 0

        # Update lap time and check if we finished a lap
        new_lap_elapsed = self.nds.memory.read_ram_u32(
            self._read_mem_chain(RAM_ADDRESSES_CHAIN["lap_time_elapsed"])
        )

        lap_changed = new_lap_elapsed < self.lap_elapsed
        self.win = self.current_lap == 3 and lap_changed

        # We terminate when the agent finishes the race (3 * lap) or when the agent hasn't progressed through the track
        terminated = self.win or (
            self.terminate_on_stall and self.last_max_distance > self.distance_timeout
        )
        reward = self._get_reward(new_distance)
        observation = self._get_obs()
        info = self._get_info()

        self.distance = new_distance
        self.lap_elapsed = new_lap_elapsed
        self.current_lap = self.nds.memory.read_ram_u32(
            self._read_mem_chain(RAM_ADDRESSES_CHAIN["lap"])
        )

        # Check if we've reached the checkpoint
        if lap_changed or new_distance >= self.last_checkpoint + self.checkpoint_size:
            self.steps_in_checkpoint = 0
            self.last_checkpoint = new_distance
        else:
            self.steps_in_checkpoint += 1

        # Render the game to window if render mode is set to human
        if self.render_mode == "human":
            self.nds.render()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def close(self):
        if self.window_opened:
            self.nds.close_window()

    def _get_obs(self):
        top_frame, _ = self.nds.get_frame()
        return top_frame

    def _get_info(self):
        return {
            "distance": self.nds.memory.read_ram_i32(
                self._read_mem_chain(RAM_ADDRESSES_CHAIN["back_distance"])
            ),
            "speed": self.nds.memory.read_ram_i32(
                self._read_mem_chain(RAM_ADDRESSES_CHAIN["speed"])
            ),
            "total_time_elapsed": self.nds.memory.read_ram_u32(
                self._read_mem_chain(RAM_ADDRESSES_CHAIN["total_time_elapsed"])
            ),
            "lap_time_elapsed": self.nds.memory.read_ram_u32(
                self._read_mem_chain(RAM_ADDRESSES_CHAIN["lap_time_elapsed"])
            ),
            "lap": self.nds.memory.read_ram_u32(
                self._read_mem_chain(RAM_ADDRESSES_CHAIN["lap"])
            ),
            "win": self.win,
        }

    def _get_reward(self, new_distance: int):
        dist_weight, speed_weight, checkpoint_weight = self.rewards_weight

        # Get each factor's reward value
        dist_reward = self._get_dist_reward(new_distance)
        speed_reward = self._get_speed_reward(new_distance)
        checkpoint_reward = self._get_checkpoint_reward(new_distance)

        # Take a weighted of the rewards using the weight values
        reward = (
            dist_weight * dist_reward
            + speed_weight * speed_reward
            + checkpoint_weight * checkpoint_reward
        )

        # Penalize the agent if it's stuck (if self.terminate_on_stall is not available)
        if not self.terminate_on_stall and (
            self.last_max_distance > self.distance_timeout
        ):
            reward -= 0.01

        return reward

    def _get_dist_reward(self, new_distance: int):
        reward_clipped = np.clip(new_distance - self.distance, -10, 10) # Clip reward to gaurd againt unexpected change
        reward_normalized = reward_clipped / 100 # Normalize the reward to a sensible value for training
        return float(reward_normalized)

    def _get_checkpoint_reward(self, new_distance: int):
        # Reward on checkpoint based on the amount of steps taken
        reward = 0
        if new_distance >= self.last_checkpoint + self.checkpoint_size:
            reward = 50 / (self.steps_in_checkpoint + 5)

        return reward

    def _get_speed_reward(self, new_distance: int):
        # Reward based on the absolute value of speed in the current direction
        MAX_SPEED = 2**16
        current_speed = self.nds.memory.read_ram_i32(
            self._read_mem_chain(RAM_ADDRESSES_CHAIN["speed"])
        )
        distance = new_distance - self.distance

        # Calculate speed per distance (so we don't accidently encourage it to take longer routes just for longer rewards)
        reward = distance * np.abs(current_speed)
        reward_normalized = reward / (MAX_SPEED * 50)

        return float(reward_normalized)

    def _render_frame(self):
        top_frame, bottom_frame = self.nds.get_frame()
        merged = np.vstack((top_frame[:, :, :3], bottom_frame[:, :, :3]))
        return merged

    def _read_mem_chain(self, chain: tuple[int]) -> int:
        cur_adr = chain[0]
        for offset in chain[1:]:
            cur_adr = self.nds.memory.read_ram_u32(cur_adr) - 0x02000000 + offset
        return cur_adr


gym.envs.register(
    id="MarioKartDS-v0", entry_point=MarioKartDSEnv, max_episode_steps=10000
)

if __name__ == "__main__":
    # env = MarioKartDSEnv("files/rom.nds", "savestates/time_trail_begining.noo", render_mode="human")
    env = gym.make(
        "MarioKartDS-v0",
        rom_path="files/rom.nds",
        savestates="savestates/time_trail_begining.noo",
        render_mode="rgb_array",
    )

    env = gym.wrappers.RecordVideo(
        env,
        video_folder="./videos",
        episode_trigger=lambda episode_id: True,  # record every episode
    )
    obs, info = env.reset()

    done = False
    i = 0
    while not done:
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        i += 1
        done = terminated | truncated
        # print(i)
        # print(reward)
    env.close()
