import argparse

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv, WarpFrame
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack

import mkds  # noqa: F401 — registers MarioKartDS-v0 in gym's registry


def make_env():
    env = gym.make(
        "MarioKartDS-v0",
        rom_path="files/rom.nds",
        savestates="savestates/time_trail_begining.noo",
        render_mode="human",
    )

    env = WarpFrame(env)
    env = MaxAndSkipEnv(env)
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to the PPO model zip")
    args = parser.parse_args()

    env = make_vec_env(make_env, 1)
    env = VecFrameStack(env, n_stack=4)

    model = PPO.load(args.path)

    obs = env.reset()
    done = False

    while not done:
        action = model.predict(obs)[0]
        obs, _reward, terminated, _info = env.step(action)
        done = terminated


if __name__ == "__main__":
    main()
