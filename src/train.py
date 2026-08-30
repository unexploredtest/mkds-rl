import os
from collections import deque

import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common import results_plotter
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv, WarpFrame
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EveryNTimesteps,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack

import mkds  # noqa: F401 — registers MarioKartDS-v0 in gym's registry
from utils import RecordActions, record_episode_actions, record_episode_policy

videos_dir = "./videos/"

# Callback for adding extra logs for total_time_elapsed, win_rate, avg_distance
# and best_finish_time (and its corresponding video)
class ExtraLogCallback(BaseCallback):
    def __init__(self, window: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.window = window
        self.wins: deque[float] = deque(maxlen=window)
        self.distances: deque[float] = deque(maxlen=window)
        self.finish_times: deque[float | None] = deque(maxlen=window)
        self.best_finish_time: float | None = None

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                won = info.get("win")
                distance = info.get("distance")
                self.wins.append(1.0 if won else 0.0)
                self.distances.append(distance)
                # One entry per race (None for losses) so this deque evicts in
                # lockstep with self.wins and never covers races older than its
                # first entry.
                finish_time_secs = info["total_time_elapsed"] / 60
                self.finish_times.append(finish_time_secs if won else None)
                if won and (
                    self.best_finish_time is None
                    or self.best_finish_time > finish_time_secs
                ):
                    self.best_finish_time = finish_time_secs

                    # Record best race
                    actions = info.get("actions")
                    record_env = make_env(preprocess=False)

                    record_episode_actions(
                        record_env, actions, videos_dir + "best_race.mp4"
                    )
                    record_env.close()

        if self.wins:
            self.logger.record("rollout/win_rate", sum(self.wins) / len(self.wins))

        if self.distances:
            self.logger.record(
                "rollout/avg_distance", sum(self.distances) / len(self.distances)
            )

        # We compute the average only for the races that were finished
        won_times = [t for t in self.finish_times if t is not None]
        if won_times:
            self.logger.record(
                "rollout/avg_finish_time",
                sum(won_times) / len(won_times),
            )

        if self.best_finish_time is not None:
            self.logger.record("rollout/best_finish_time", self.best_finish_time)

        return True

# Callback for recording a race with the current model
class RecordRaceCallback(BaseCallback):
    def _on_step(self) -> bool:
        # Fresh env matching the training obs pipeline so the policy can predict
        record_env = VecFrameStack(make_vec_env(make_env, 1), n_stack=4)
        record_episode_policy(
            record_env, self.model, videos_dir + f"race_{self.num_timesteps}.mp4"
        )
        record_env.close()
        return True

# Finds savestates listed in folder_path with extension ".noo"
def find_state_files(folder_path):
    state_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".noo"):
                state_files.append(os.path.join(root, file))
    return state_files

# A function to make a default environment
def make_env(preprocess=True):
    rom_path = "files/rom.nds"
    savestates = find_state_files("savestates/")
    env = gym.make(
        "MarioKartDS-v0",
        rom_path=rom_path,
        savestates=savestates,
        render_mode="rgb_array",
    )
    env = RecordActions(env)
    if preprocess:
        env = WarpFrame(env)
        env = MaxAndSkipEnv(env)
    return env


if __name__ == "__main__":
    # Constants
    num_cpu = 4
    training_steps = 20_000_000
    saving_freq = 200_000
    log_dir = "./logs/"
    models_dir = "./models/"
    plots_dir = "./plots/"
    tensorboard_log = "./mariokartds_tensorboard/"

    # Making an environment (num_cpu amount) 
    env = make_vec_env(make_env, num_cpu, monitor_dir=log_dir)
    env = VecFrameStack(env, n_stack=4)

    model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=tensorboard_log)

    # Making checkpoints
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(videos_dir, exist_ok=True)

    # Adding callbacks
    checkpoint_on_event = CheckpointCallback(save_freq=1, save_path="./models/")
    event_callback = EveryNTimesteps(n_steps=saving_freq, callback=checkpoint_on_event)

    win_rate_callback = ExtraLogCallback(window=100)

    record_race_callback = RecordRaceCallback()
    record_race_event = EveryNTimesteps(n_steps=200_000, callback=record_race_callback)

    # Learn for training_steps
    model.learn(
        total_timesteps=training_steps,
        callback=CallbackList([event_callback, win_rate_callback, record_race_event]),
    )

    # We close the environment to ensure that the emulator is closed
    env.close()

    # plot results
    results_plotter.plot_results(
        [log_dir], 1e5, results_plotter.X_TIMESTEPS, "PPO Mario Kart DS"
    )

    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(plots_dir + "plot.png")
    plt.close()
