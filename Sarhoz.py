import gymnasium as gym
import sys
import highway_env_copy as env
from matplotlib import pyplot as plt
import pprint
from stable_baselines3 import DQN
import cv2
from stable_baselines3 import PPO
from sb3_contrib import TRPO

# Video
frameSize = (1280,560)
out = cv2.VideoWriter('video'+"-Merging"+'.avi', cv2.VideoWriter_fourcc(*'mp4v'), 16, frameSize)

def model_creation(model_name: str):
    env = gym.make("merge-in-v0")

    if (model_name == "DQN"):
        print("DQN")
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
              tensorboard_log="highway_DQN/",
              device='cuda')
        model.learn(20000)
        model.save("highway_dqn/model")
    elif (model_name == "PPO"):
        print("PPO")
        model = PPO("MlpPolicy", env,
                verbose=1,
                learning_rate=0.0001,
                batch_size=32,
                gamma=0.8,
                tensorboard_log="highway_PPO/",
                device='cuda')
        model.learn(20000)
        model.save("highway_ppo/model")
    elif (model_name == "TRPO"):
        print("TRPO")
        model = model = TRPO("MlpPolicy", env, 
             verbose=1,
             learning_rate=0.0001,
             batch_size=32,
             gamma=0.8,
             tensorboard_log="highway_TRPO/",
             device='cuda')
        model.learn(20000)
        model.save("highway_trpo/model")
    else:
        print("Input model does not exist!")


def DRL_Models():
    
    # Train model 
    #model_creation("DQN")
    #model_creation("PPO")
    #model_creation("TRPO")

    # Load model
    #model = DQN.load("highway_dqn/model") #--> 12 colisions but looks really weird when merging
    #model = PPO.load("highway_ppo/model") #--> 12 colisions
    #model = TRPO.load("highway_trpo/model") #--> 10 colisions

    env = gym.make('merge-in-v0', render_mode='rgb_array')
    #env = gym.make('intersection-v1', render_mode='rgb_array')
    env.configure({
    "screen_width": 1280,
    "screen_height": 560,
    "renderfps": 16
    })

    pprint.pprint(env.config)


    # create a image
    number_of_collisions = 0
    T = 0
    while T <= 100:
    
        done = truncated = False
        obs, info = env.reset()
    
        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            if info.get('crashed'):
                number_of_collisions += 1
            env.render()
            cur_frame = env.render()
            out.write(cur_frame)
        T+=1
    print('crashrate is '+str(float(number_of_collisions)/T)+' and T is'+ str(T))
    print('number_of_collisions is:', number_of_collisions)
    print('DONE')
    out.release()

DRL_Models()