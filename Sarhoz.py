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
                tensorboard_log="highway_dqn/")
        model.learn(3000)
        model.save("highway_dqn/model")
    elif (model_name == "PPO"):
        print("PPO")
        model = PPO("MlpPolicy", env,
                    learning_rate=0.0003)
        model.learn(3000)
        model.save("highway_ppo/model")
    elif (model_name == "TRPO"):
        print("TRPO")
        model = model = TRPO("MlpPolicy", env,
             learning_rate=0.0003,
             n_steps=1024,
             batch_size=128,
             gamma=0.99,
             cg_max_steps=15,
             cg_damping=0.1,
             line_search_shrinking_factor=0.8,
             line_search_max_iter=10,
             n_critic_updates=10,
             gae_lambda=0.95,
             use_sde=False,
             sde_sample_freq=-1,
             normalize_advantage=True,
             target_kl=0.015,
             sub_sampling_factor=1,
             policy_kwargs=None,
             verbose=1,
             tensorboard_log="highway_TRPO/",
             seed=None,
             device='cuda',
             _init_setup_model=True)
        model.learn(3000)
        model.save("highway_trpo/model")
    else:
        print("Input model does not exist!")


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

def DRL_Models():
    
    # Train model 
    #model_creation("DQN")

    # Load model
    model = DQN.load("highway_dqn/model")
    # model = PPO.load("highway_ppo/model")
    # model = TPRO.load("highway_trpo/model")

    env = gym.make('merge-in-v0', render_mode='rgb_array')
    #env = gym.make('racetrack-v0')
    env.configure({
    "screen_width": 1280,
    "screen_height": 560,
    "renderfps": 16
    })
    # env.configure({
    # "action": {
    #     "type": "ContinuousAction"
    # },
    # "offroad_terminal": False,
    # "other_vehicles": 1,
    # "vehicles_count": 6,
    # "initial_vehicle_count": 0,
    # "spawn_probability": 0.
    # })

    #pprint.pprint(env.config)


    # create a image
    number_of_collisions = 0
    T = 0
    while T <= 100:
    
        done = truncated = False
        obs, info = env.reset()
    
        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            print(action)
            obs, reward, done, truncated, info = env.step(action)
            print(info)
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