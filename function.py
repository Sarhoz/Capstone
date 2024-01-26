import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3 import PPO
from sb3_contrib import TRPO
from highway_env_copy.vehicle.kinematics import Performance, Logger
from matplotlib import pyplot as plt
import cv2
import pprint
import warnings
warnings.filterwarnings("ignore")

# Created environment
def environment(environment_name: str, Action_type: bool):
    env = gym.make(environment_name, render_mode = "rgb_array")
    
    env.configure({
    "screen_width": 1280,
    "screen_height": 560,
    "renderfps": 16})

    # The action type is true ==> ContinuousActionSpace
    if Action_type:
        env.configure({"action": {
                "type": "ContinuousAction",
                "longitudinal": True,
                "lateral": True,
                "target_speeds": [20, 30]
                }})

    pprint.pprint(env.config)
    
    env.reset()
    
    return env

# Training/loading baseline models (with hyperparameters tuned)
# No tuning models created by commenting learning_rate, batch_size, gamma ---> tuning done by uncommenting these parameters
# These files for merging V0 (none tuning and tuning) created with this function (by changing the logging names)
# Also used this for merging V3 baseline (no tuning)
def baseline_models(model: str, env, iterations: int, rewards: bool):
    if model.upper() == "TRPO":
        model = TRPO("MlpPolicy", env,
                     tensorboard_log="Tensorboard_log/Merging_v3_model_TRPO",
                     device="cuda",
                     learning_rate= 0.000103,
                     batch_size= 32,
                     gamma= 0.999,
                     verbose=1)
        model.learn(iterations, progress_bar=True)
        print(f"{model} has finished training with {iterations} iterations!")
        if rewards:
            model.save("Training models/highway_trpo/Merging_v3_model_TRPO")
        else:
            model.save("Training models/highway_trpo/Merging_v0_model_TRPO")
        return model
    elif model.upper() == "PPO":
        model = PPO("MlpPolicy", env,
                     tensorboard_log="Tensorboard_log/Merging_v3_model_PPO",
                     device="cuda",
                     learning_rate=0.00015,
                     batch_size=32,
                     gamma=0.99,
                     verbose=1)
        model.learn(iterations, progress_bar=True)
        print(f"{model} has finished training with {iterations} iterations!")
        if rewards:
            model.save("Training models/highway_ppo/Merging_v3_model_PPO")
        else:
            model.save("Training models/highway_ppo/Merging_v0_model_PPO")
        return model
    elif model.upper() == "DQN":
        model = DQN("MlpPolicy", env,
                     tensorboard_log="Tensorboard_log/Merging_v3_model_DQN",
                     device="cuda",
                     gamma=0.99,
                     learning_rate=0.0043,
                     batch_size=512,
                     verbose=1)
        model.learn(iterations, progress_bar=True)
        print(f"{model} has finished training with {iterations} iterations!")
        if rewards:
            model.save("Training models/highway_dqn/Merging_v3_model_DQN")
        else:
            model.save("Training models/highway_dqn/Merging_v0_model_DQN")
        return model
    else:
        print("The input algorithm does not exist!")

# Function made for tuning models of modified reward system
def tuned_reward_models(model: str, env, iterations: int):
    if model.upper() == "TRPO":
        model = TRPO("MlpPolicy", env,
                     tensorboard_log="Tensorboard_log/Merging_v3_model_Tuned_TRPO",
                     device="cuda",
                     gamma=0.95,
                     learning_rate=3.846887458592866e-05,
                     batch_size=128,
                     verbose=1)
        model.learn(iterations, progress_bar=True)
        print(f"{model} has finished training with {iterations} iterations!")
        model.save("Training models/highway_trpo/Merging_v3_model_Tuned_TRPO")
        return model
    elif model.upper() == "PPO":
        model = PPO("MlpPolicy", env,
                     tensorboard_log="Tensorboard_log/Merging_v3_model_Tuned_PPO",
                     device="cuda",
                     learning_rate=3.6931978466449305e-05,
                     batch_size=128,
                     gamma=0.995,
                     verbose=1)
        model.learn(iterations, progress_bar=True)
        print(f"{model} has finished training with {iterations} iterations!")
        model.save("Training models/highway_ppo/Merging_v3_model_Tuned_PPO")
        return model
    elif model.upper() == "DQN":
        model = DQN("MlpPolicy", env,
                     tensorboard_log="Tensorboard_log/Merging_v3_model_Tuned_DQN",
                     device="cuda",
                     gamma=0.95,
                     learning_rate=0.00072,
                     batch_size=64,
                     verbose=1)
        model.learn(iterations, progress_bar=True)
        print(f"{model} has finished training with {iterations} iterations!")
        model.save("Training models/highway_dqn/Merging_v3_model_Tuned_DQN")
        return model
    else:
        print("The input algorithm does not exist!")

# Check the Performance of a model
def performance_model(env, model, model_name: str, model_path: str, number_of_tests: int, video_name:str, i: int, base_reward: bool):

    # Video
    frameSize = (1280,560)
    out = cv2.VideoWriter('video'+"-Merging-"+ video_name + '.avi', cv2.VideoWriter_fourcc(*'mp4v'), 16, frameSize)
    all_ttc = []
    # load model
    model = model.load(model_path)
    
    # Performance and logger
    perfm = Performance()
    lolly = Logger()

    # Test run
    number_of_collisions = 0
    T = 0
    best_reward = -float('inf') # initialize the best reward with negative infinity
    rewards = [] #initialize list of rewards
    for i in range(number_of_tests):
    
        done = truncated = False
        obs, info = env.reset()
        reward = 0

        ego_car = env.controlled_vehicles[0]

        total_reward = 0 # total reward for this epoch

        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            #print("info: ", info)
            #print("Reward in functions: ", reward)

            # Only safe the important moments as the large numbers are not important!
            if info['TTC'] <= 20:
                all_ttc.append(info['TTC'])
                print("TTC under 20: ", info['TTC'])
            

            lolly.file(ego_car)
            if info.get('crashed'):
                number_of_collisions += 1

            env.render()
            cur_frame = env.render()
            out.write(cur_frame)

        if total_reward > best_reward:
            best_reward = total_reward

        rewards.append(total_reward)
        #print("total reward: ", total_reward)
        T+=1
        print(T)
        perfm.add_measurement(lolly)
        lolly.clear_log()
    
    plt.plot(rewards)
    plt.title("Rewards per run")
    plt.xlabel('Runs')
    plt.ylabel("Total Reward")
    plt.show()

    perfm.print_performance()
    
    print(f'Best Reward: {best_reward}') # print best reward
    print('crashrate is '+ str(float(number_of_collisions)/T) +' and T is '+ str(T))
    print('number_of_collisions is: ', number_of_collisions)
    print(f"minimum ttc: {min(all_ttc)}")
    print(f"average ttc: {sum(all_ttc) / len(all_ttc)}")
    writing = "w" if i == 0 else "a"

    if (base_reward):
        temp = "baseline rewards"
    else:
        temp = "modified rewards"

    with open("Performance.txt", writing) as file:
        file.write(f"\n The {model_name} model with {temp} (DiscreteMeteAction) \n \n")
        file.write(f"{perfm.string_rep()}")
        file.write(f"\n The average TTC of {T} measurements is: {(sum(all_ttc) / len(all_ttc))}")
        file.write(f"\n The minimum TTC of {T} measurements is: {min(all_ttc)} \n")
        file.write(f"\n\n")

    print('DONE')
    out.release()

    return perfm.array_rep()


