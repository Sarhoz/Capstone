import functions
from functions.utils import environment, baseline_models, performance_model, tuned_reward_models
# import highway_env_copy as env
from matplotlib import pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3 import PPO
from sb3_contrib import TRPO
import joblib


# Create Environment
env = environment("merge-in-v3", False)

# Creating the Baseline Models ---> Used for requirement 1 and requirement 2
# baseline_models("DQN", env, 50000, True)
# baseline_models("PPO", env, 50000, True)
# baseline_models("TRPO", env, 50000, True)

# Creating the Reward models with tuning ---> requirement 2
# tuned_reward_models("DQN",env, 50000)
# tuned_reward_models("PPO",env, 50000)
#tuned_reward_models("TRPO",env, 100000)

# Creating the best model
# ...

# Look at the Performance
# number_of_test has to be larger than 1.
# performance = performance_model(env=env, model=TRPO, model_path="Training models\highway_TRPO\Merging_v3_model_Tuned_TRPO.zip", model_name="TRPO",
#                                 number_of_tests=500, video_name="Merging_v3_TRPO_Modified_Rewards_Tuned", i=4, base_reward=False)

# Look at parameters of models
# study = joblib.load("models/PPO/merge_in_v3/tuning/run_3/study.pkl")
# print("Best trial until now:")
# print(" Value: ", study.best_trial.value)
# print(" Params: ")
# for key, value in study.best_trial.params.items():
#     print(f"    {key}: {value}")
