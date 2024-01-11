import gymnasium as gym
import highway_env as env
from matplotlib import pyplot as plt
import pprint
from stable_baselines3 import DQN
import cv2

# Video
frameSize = (1280,560)
out = cv2.VideoWriter('video'+"-Merging"+'.avi', cv2.VideoWriter_fourcc(*'mp4v'), 16, frameSize)

def DQN_Creation():
    env = gym.make("merge-in-v0")
    model = DQN('MlpPolicy', env,
              policy_kwargs=dict(net_arch=[256, 256]),
              learning_rate=5e-4,
              buffer_size=15000,
              learning_starts=200,
              batch_size=32,
              gamma=0.8,
              train_freq=1,
              gradient_steps=1,
              target_update_interval=50,
              verbose=1,
              tensorboard_log="highway_dqn/")
    model.learn(30000)
    model.save("highway_dqn/model")

# Unlearned Agent
def Unlearned_Agent():
    env = gym.make('merge-in-v0', render_mode='rgb_array')
    env.configure({
    "screen_width": 1280,
    "screen_height": 560,
    "renderfps": 16
    })

    # Looking at data of the env
    pprint.pprint(env.config)
    obs = env.reset()
    print(obs)
    print(env.get_available_actions())

    # create a image
    done = truncated = False
    obs, info = env.reset()
    
    while not (done or truncated):
        action = env.action_type.actions_indexes["IDLE"]
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        if info.get('crashed'):
            number_of_collisions += 1
        env.render()
        cur_frame = env.render()
        out.write(cur_frame)
    out.release()

def DQN_Agent():


    # Train model 
    #DQN_Creation()

    # Load model
    model = DQN.load("highway_dqn/model")

    env = gym.make('merge-in-v0', render_mode='rgb_array')
    env.configure({
    "screen_width": 1280,
    "screen_height": 560,
    "renderfps": 16
    })

    # create a image
    done = truncated = False
    obs, info = env.reset()
    
    while not (done or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        cur_frame = env.render()
        out.write(cur_frame)
    out.release()

DQN_Agent()