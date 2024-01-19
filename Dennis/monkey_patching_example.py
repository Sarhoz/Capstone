# import highway_env
# from utils import MyVehicle, monkey_patcher
# highway_env.vehicle.kinematics.Vehicle = monkey_patcher(highway_env.vehicle.kinematics.Vehicle, MyVehicle)

# import gymnasium as gym
# import merge_in_env



# # Create new environment
# gym.register(id="merge-in-v0", entry_point="merge_in_env:MergeinEnv")
# env = gym.make("merge-in-v0", render_mode="rgb_array")
# env.reset()
# # Extract the agent from the environment
# ego_car = env.unwrapped.controlled_vehicles[0]
# print(ego_car.crashed)
# # Check to see if patching worked
# try:
#     print(ego_car.track_affiliated_lane)
# except:
#     print(f"Class Object still does not exist")
# finally:
#     env.close()
#     del env, ego_car





