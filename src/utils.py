import gymnasium as gym
import imageio.v2 as imageio


class RecordActions(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # self.prev_actions = []
        self.actions = []

    def reset(self, **kwargs):
        # self.prev_actions = self.actions.copy()
        self.actions = []
        return self.env.reset(**kwargs)

    def step(self, action):
        self.actions.append(action)

        observation, reward, terminated, truncated, info = self.env.step(action)
        # info["prev_actions"] = self.prev_actions
        info["actions"] = self.actions
        return observation, reward, terminated, truncated, info


def record_episode_policy(env, model, video_name):
    with imageio.get_writer(video_name, fps=15, codec="libx264") as writer:
        obs = env.reset()

        done = False
        while not done:
            action = model.predict(obs)[0]
            obs, reward, terminated, info = env.step(action)

            frame = env.venv.envs[0].unwrapped.render()
            writer.append_data(frame)

            done = terminated


def record_episode_actions(env, actions, video_name):
    with imageio.get_writer(video_name, fps=60, codec="libx264") as writer:
        obs, info = env.reset()

        done = False
        i = 0
        while not done:
            action = actions[i % len(actions)]
            obs, reward, terminated, truncated, info = env.step(action)

            frame = env.unwrapped.render()
            writer.append_data(frame)

            done = terminated or truncated
            i += 1
