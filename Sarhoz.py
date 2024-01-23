# import highway_env_copy as env
from matplotlib import pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3 import PPO
from sb3_contrib import TRPO
import joblib

from function import environment, baseline_models, performance_model


# Create Environment
env = environment("merge-in-v3", False)

# Creating the Baseline Models ---> Used for requirement 1 and requirement 2
#baseline_models("DQN", env, 50000, False)
#baseline_models("PPO", env, 50000, True)
baseline_models("TRPO", env, 20000, True)

# Creating the Reward models with tuning ---> requirement 3
#Tuned_reward_models("DQN", 50000)
#Tuned_reward_models("PPO", 50000)
#Tuned_reward_models("TRPO", 50000)

# Creating the best model
# ...

# Look at the Performance
performance = performance_model(env=env, model=TRPO, model_path="Training models\highway_TRPO\Merging_v3_model_TRPO.zip", model_name="TRPO",
                              number_of_tests=500, video_name="Merging_v3_TRPO_Base_Rewards", i=4, base_reward=True)

# Look at parameters of models
# study = joblib.load("models/TRPO/merge_in_v0/tuning/run_2/study.pkl")
# print("Best trial until now:")
# print(" Value: ", study.best_trial.value)
# print(" Params: ")
# for key, value in study.best_trial.params.items():
#     print(f"    {key}: {value}")
