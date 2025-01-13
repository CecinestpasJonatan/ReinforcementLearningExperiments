import torch as th
import torch.nn as nn

import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

def linear_schedule(initial_value, power=2):
    def func(progress_remaining):
        return (progress_remaining**power) * initial_value
    return func

vec_env = gym.make('Hopper-v5')
vec_env = make_vec_env(vec_env, n_envs=8)

policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=dict(pi=[512, 512], vf=[512]))

model = PPO("MlpPolicy", vec_env, verbose=1, device="cpu", learning_rate=linear_schedule(0.001, ))
model.learn(total_timesteps=250000)

while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")