# import highway_env_copy as env
from matplotlib import pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3 import PPO
from sb3_contrib import TRPO


from function import environment, baseline_models, performance_model


# Create Environment
env = environment("merge-in-v0")

# Creating the Baseline Models
#baseline_models("DQN", env, 80000, False)
#baseline_models("PPO", env, 50000, False)
#baseline_models("TRPO", env, 50000, False)

# Look at the Performance
performance = performance_model(env=env, model=DQN, model_path="highway_dqn/model-baseline", model_name="DQN",
                                number_of_tests=10, video_name="baseline_test", i=2, base_reward=True)
