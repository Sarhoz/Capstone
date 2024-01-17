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

from function import environment, baseline_models, performance_model


# Create Environment
env = environment("merge-in-v0")

# Creating the Baseline Models
#baseline_models("DQN", env, 80000, False)
#baseline_models("PPO", env, 50000, False)
#baseline_models("TRPO", env, 50000, False)

# Look at the Performance
performance = performance_model(env, PPO, "highway_ppo/model-baseline", 50, "baseline_test", i=2)