import gymnasium as gym
import highway_env as env
from matplotlib import pyplot as plt
import highway_env
highway_env.register_highway_envs()

env = gym.make('merge-in-v0', render_mode='rgb_array')

env.configure({
    "screen_width": 1920,
    "screen_height": 1080,
    "scaling": 10,
    "renderfps": 15
})
env.configure({
    "show_trajectories": False,
    "manual_control": True
})

env.reset()
for _ in range(300):
    action = env.action_type.actions_indexes["IDLE"]
    obs, reward, done, truncated, info = env.step(action)
    env.render()
plt.imshow(env.render())
plt.show()
