import torch as th
import torch.nn as nn

import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

def linear_schedule(initial_value):
    def func(progress_remaining):
        return progress_remaining * initial_value
    return func

# Parallel environments
vec_env = gym.make('Hopper-v5', render_mode='human')

policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=dict(pi=[512, 512], vf=[512, 512]))

model = PPO("MlpPolicy", vec_env, verbose=1, device="cpu", learning_rate=linear_schedule(0.001))
model.learn(total_timesteps=250000)
# model.save("ppo_cartpole")

# del model # remove to demonstrate saving and loading

# model = PPO.load("ppo_cartpole")

# obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")