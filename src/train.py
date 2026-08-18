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


class WinRateCallback(BaseCallback):
    """Logs the rolling win rate from ``info["win"]`` over the last N episodes."""

    def __init__(self, window: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.window = window
        self.wins: deque[float] = deque(maxlen=window)

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                self.wins.append(1.0 if info.get("win") else 0.0)
        if self.wins:
            self.logger.record("rollout/win_rate", sum(self.wins) / len(self.wins))
        return True


def find_state_files(folder_path):
    state_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".noo"):
                state_files.append(os.path.join(root, file))
    return state_files


def make_env():
    rom_path = "files/rom.nds"
    savestates = find_state_files("savestates/")
    env = gym.make("MarioKartDS-v0", rom_path=rom_path, savestates=savestates)
    env = WarpFrame(env)
    env = MaxAndSkipEnv(env)
    return env


if __name__ == "__main__":
    num_cpu = 4
    training_steps = 20_000_000
    saving_freq = 200_000
    log_dir = "./logs/"
    models_dir = "./models/"
    plots_dir = "./plots/"
    tensorboard_log = "./mariokartds_tensorboard/"

    env = make_vec_env(make_env, num_cpu, monitor_dir=log_dir)
    env = VecFrameStack(env, n_stack=4)

    model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=tensorboard_log)

    # Making checkpoints
    os.makedirs(models_dir, exist_ok=True)
    checkpoint_on_event = CheckpointCallback(save_freq=1, save_path="./models/")
    event_callback = EveryNTimesteps(n_steps=saving_freq, callback=checkpoint_on_event)

    win_rate_callback = WinRateCallback(window=100)

    model.learn(
        total_timesteps=training_steps,
        callback=CallbackList([event_callback, win_rate_callback]),
    )

    env.close()

    # plot results
    results_plotter.plot_results(
        [log_dir], 1e5, results_plotter.X_TIMESTEPS, "PPO Mario Kart DS"
    )

    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(plots_dir + "plot.png")
    plt.close()
