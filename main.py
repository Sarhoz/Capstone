import functions
from functions.utils import environment, baseline_models, performance_model, tuned_reward_models, best_model
# import highway_env_copy as env
from matplotlib import pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3 import PPO
from sb3_contrib import TRPO
import joblib


# Create Environment
env = environment("merge-in-v3", False)

# Creating/learning/saving models (all of the below)
# ---- used for baseline rewards (with and without tuning) / modified rewards with no tuning) ----
# baseline_models("DQN", env, 50000, True)
# baseline_models("PPO", env, 50000, True)
# baseline_models("TRPO", env, 50000, True)

# ---- used for modified rewards tuning ----
# tuned_reward_models("DQN",env, 50000)
# tuned_reward_models("PPO",env, 100000)
# tuned_reward_models("TRPO",env, 100000)

# ---- The best possible model (not yet implemented; we are waiting for the tuning to be done as that takes days!)
# best_model(emv, 100000)

# Get the performance of a certain trained model ---> returns a video of the car merging and the performance of the car of the different situations
# ---- number_of_test has to be larger than 1 ----
# ---- To understand the code look inside functions.utils ----
performance = performance_model(env=env, model=DQN, model_path="Training models\highway_DQN\Merging_v3_model_Tuned_DQN.zip", model_name="DQN",
                                 number_of_tests=500, video_name="Merging_v3_DQN_Modified_Rewards_Tuned", i=4, base_reward=False)

# The code below prints the best hyperparameters after tuning
# study = joblib.load("models/PPO/merge_in_v3/tuning/run_3/study.pkl")
# print("Best trial until now:")
# print(" Value: ", study.best_trial.value)
# print(" Params: ")
# for key, value in study.best_trial.params.items():
#     print(f"    {key}: {value}")
