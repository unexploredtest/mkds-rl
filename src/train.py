import os

import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EveryNTimesteps
from stable_baselines3.common import results_plotter

from mkds import MarioKartEnv

def find_state_files(folder_path):
    state_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.noo'):
                state_files.append(os.path.join(root, file))
    return state_files

def make_env():
    rom_path = "files/rom.nds"
    savestates = find_state_files("savestates/")
    env = MarioKartEnv(rom_path, savestates)
    env = gym.wrappers.TimeLimit(env, 1000)
    env.reset()
    return env

if __name__ == "__main__":
    num_cpu = 4
    training_steps = 10_000_000
    saving_freq = 50_000
    log_dir = "./logs/"
    models_dir = "./models/"
    plots_dir = "./plots/"

    env = make_vec_env(make_env, num_cpu, monitor_dir=log_dir)

    model = PPO("CnnPolicy", env, verbose=1)

    # Making checkpoints
    os.makedirs(models_dir, exist_ok=True)
    checkpoint_on_event = CheckpointCallback(save_freq=1, save_path="./models/")
    event_callback = EveryNTimesteps(n_steps=saving_freq, callback=checkpoint_on_event)

    model.learn(total_timesteps=training_steps, callback=event_callback)

    env.close()

    # plot results
    results_plotter.plot_results(
        [log_dir], 1e5, results_plotter.X_TIMESTEPS, "PPO Mario Kart DS"
    )

    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(plots_dir + "plot.png")
    plt.close()