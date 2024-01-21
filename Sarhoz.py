# import highway_env_copy as env
from matplotlib import pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3 import PPO
from sb3_contrib import TRPO
import joblib

from function import environment, baseline_models, performance_model


# Create Environment
env = environment("merge-in-v0", False)

# Creating the Baseline Models
#baseline_models("DQN", env, 80000, False)
#baseline_models("PPO", env, 50000, False)
#baseline_models("TRPO", env, 50000, True)

# Look at the Performance
# performance = performance_model(env=env, model=TRPO, model_path="highway_TRPO\model-baseline_TTC.zip", model_name="TRPO",
#                                number_of_tests=5, video_name="baseline_test", i=4, base_reward=True)


# Look at parameters of models
# study = joblib.load("models/PPO/merge_in_v0/tuning/run_2/study.pkl")
# print("Best trial until now:")
# print(" Value: ", study.best_trial.value)
# print(" Params: ")
# for key, value in study.best_trial.params.items():
#     print(f"    {key}: {value}")
