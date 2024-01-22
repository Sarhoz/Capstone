import gymnasium as gym
import sys
# import highway_env_copy as env
from matplotlib import pyplot as plt
import pprint
from stable_baselines3 import DQN
import cv2
from stable_baselines3 import PPO
from sb3_contrib import TRPO
import numpy as np

from highway_env_copy.vehicle.kinematics import Performance, Logger


# Video
frameSize = (1280,560)
out = cv2.VideoWriter('video'+"-Merging"+'.avi', cv2.VideoWriter_fourcc(*'mp4v'), 16, frameSize)

# Create enviromenent
env = gym.make("merge-in-v4", render_mode = "rgb_array")
#env = gym.make("racetrack-v0", render_mode = "rgb_array")

# env.configure({
#      "action": {
#             "type": "ContinuousAction"
#             },
#     })

def model_creation(model_name: str):
    
    env.reset()
    pprint.pprint(env.config)

    if (model_name == "DQN"):
        print("DQN")
        # DQN CAN NOT LEARN CONTINUOUS!!!
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
              #tensorboard_log="highway_DQN/",
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
                #tensorboard_log="highway_PPO/",
                device='cuda')
        model.learn(50000)
        model.save("highway_ppo/model-S1")
    elif (model_name == "TRPO"):
        print("TRPO")
        model = TRPO("MlpPolicy", env,
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
             #tensorboard_log="highway_TRPO/",
             seed=None,
             device='cuda',
             _init_setup_model=True)
        model.learn(25000)
        model.save("highway_trpo/model-Salih-V3-tuning3")
    else:
        print("Input model does not exist!")


def DRL_Models():
    
    # Train model 

    #model_creation("DQN")
    #model_creation("PPO")
    #model_creation("TRPO")

    # Load model
    #model = DQN.load("highway_dqn/model")
    #model = PPO.load("highway_ppo/model-S1") 
    model = TRPO.load("Training models\highway_TRPO\model-Salih-V3-tuning2.zip")

    env.configure({
    "screen_width": 1280,
    "screen_height": 560,
    "renderfps": 16
    })

    pprint.pprint(env.config)
    env.reset()
    #Performance and logger
    perfm = Performance()
    lolly = Logger()

    
    # create a image
    number_of_collisions = 0
    T = 1
    best_reward = -float('inf') # initialize the best reward with negative infinity
    rewards = [] #initialize list of rewards
    while T <= 100:
    
        done = truncated = False
        obs, info = env.reset()
        reward = 0
        ego_car = env.controlled_vehicles[0]
        stepcounter = 0

        total_reward = 0 # total reward for this epoch

        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            stepcounter += 1
            total_reward += reward
            print(info)
            print(reward)
            lolly.file(ego_car)

            if info.get('crashed'):
                number_of_collisions += 1
            env.render()
            cur_frame = env.render()
            out.write(cur_frame)

        if total_reward > best_reward:
            best_reward = total_reward
        rewards.append(total_reward)
        T+=1
        perfm.add_measurement(lolly)
        lolly.clear_log()
        print(T)


    plt.plot(rewards)
    plt.title("Rewards per run")
    plt.xlabel('Runs')
    plt.ylabel("Total Reward")
    plt.show()

    perfm.print_performance()
    
    print(f'Best Reward: {best_reward}') # print best reward

    print('crashrate is '+str(float(number_of_collisions)/T)+' and T is'+ str(T))
    print('number_of_collisions is:', number_of_collisions)
    
    #a stands for "append"
    with open("Performance.txt", "a") as file:
        file.write(f"\n The TRPO  with Salih rewards (v3) (no Tuning and DiscreteAction) (TTC threshold 1 sec) -- 100 runs -- merge \n \n")
        file.write(f"{perfm.string_rep()}")
        file.write(f"\n")
        file.write(f"{perfm.array_rep()}")
        file.write(f"\n\n")

    print('DONE')
    out.release()

DRL_Models()

