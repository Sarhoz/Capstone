"""
Utility file containing helper functions and custom classes.
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pprint
import warnings
warnings.filterwarnings("ignore")

import gymnasium as gym

import highway_env
from highway_env.road.road import Road
from highway_env.utils import Vector
from highway_env.vehicle.kinematics import Vehicle

from stable_baselines3 import DQN, PPO
from sb3_contrib import TRPO


##############################################################
#                       CUSTOM CLASSES                       #
##############################################################

# Augmented 'Vehicle' class
class MyVehicle(Vehicle):
    
    """
    --- Augmented version of the Vehicle class from highway-env ---

    A moving vehicle on a road, and its kinematics.

    The vehicle is represented by a dynamical system: a modified bicycle model.
    It's state is propagated depending on its steering and acceleration actions.
    """

    COMFORT_ACC_MAX = 3.0
    """ Desired maximum acceleration [m/s2] """
    COMFORT_ACC_MIN = -5.0
    """ Desired maximum deceleration [m/s2] """

    def __init__(self,
                 road: Road,
                 position: Vector,
                 heading: float = 0,
                 speed: float = 0,
                 predition_type: str = 'constant_steering'):
        super().__init__(road, position, heading, speed, predition_type)
        self.recorded_actions = [{'steering': 0, 'acceleration': 0}]
        self.recorded_positions = [position.tolist()]
        self.destination_location = np.array([0,0])
        self.track_affiliated_lane = False

    def step(self, dt: float) -> None:
        """
        Propagate the vehicle state given its actions.

        Integrate a modified bicycle model with a 1st-order response on the steering wheel dynamics.
        If the vehicle is crashed, the actions are overridden with erratic steering and braking until complete stop.
        The vehicle's current lane is updated.

        :param dt: timestep of integration of the model [s]
        """
        # Copied part
        self.clip_actions()
        delta_f = self.action['steering']
        beta = np.arctan(1 / 2 * np.tan(delta_f))
        v = self.speed * np.array([np.cos(self.heading + beta),
                                   np.sin(self.heading + beta)])
        self.position += v * dt
        if self.impact is not None:
            self.position += self.impact
            self.crashed = True
            self.impact = None
        self.heading += self.speed * np.sin(beta) / (self.LENGTH / 2) * dt
        self.speed += self.action['acceleration'] * dt
        self.on_state_update()
        # Custom added part
        self.recorded_actions.append(self.action)
        self.recorded_positions.append(self.position.tolist())
    
    @property
    def lane_distance(self) -> float:
        return self.lane.distance(self.position)
    
    @property
    def lane_heading_difference(self) -> float:
        """
        Signed lane heading difference. Wrapping perserves the sign.
        
        Remark: The sign of the multiplication of lane_distance and lane_difference_heading is
        - Positive, whenever the car if deviating from the road
        - Negative, whenever the car is heading to the road
        """
        if self.lane is None:
            print('trouble coming!')
        #conditional wrapping to confine the angle
        if self.heading-self.lane.lane_heading(self.position) < -np.pi:
            return self.heading-self.lane.lane_heading(self.position)+2*np.pi
        elif self.heading-self.lane.lane_heading(self.position) > np.pi:
            return self.heading-self.lane.lane_heading(self.position)-2*np.pi
        #default
        return self.heading-self.lane.lane_heading(self.position)
        #old unsigned difference
        #return min(abs(self.heading-self.lane.lane_heading(self.position)), abs(self.lane.lane_heading(self.position) + self.heading))

    @property
    def position_change(self) -> float:
        if len(self.recorded_positions) < 2:
            return 0
        return np.linalg.norm(np.array(self.recorded_positions[-1]) - np.array(self.recorded_positions[-2]))

    @property
    def jerk(self) -> float:
        if len(self.recorded_actions) < 2:
            return 0
        jerk_accel = abs(self.recorded_actions[-2]['acceleration'] - self.recorded_actions[-1]['acceleration']) / (
                self.COMFORT_ACC_MAX - self.COMFORT_ACC_MIN)
        jerk_steer = abs(self.recorded_actions[-2]['steering'] - self.recorded_actions[-1]['steering']) * 2 / np.pi
        return (jerk_accel + jerk_steer) / 2
    
    @property
    def destination_key(self) -> np.ndarray:
        if getattr(self, "route", None):
            last_lane_index = self.route[-1]
            last_lane_index = last_lane_index if last_lane_index[-1] is not None else (*last_lane_index[:-1], 0)
            return last_lane_index
        else:
            return self.position


# Logger class
class Logger:
    def __init__(self):
        self.speed = []  # float
        self.collision = []  # boolean
        self.travel_distance = []  # float
        # self.ttc = [] #float
        # self.right_lane = [] #float
        self.duration = 0  # len(self.speed)

    def clear_log(self):
        self.__init__()

    def file(self, v: Vehicle):
        self.speed.append(v.speed)
        self.collision.append(v.crashed)
        self.travel_distance.append(
            v.position_change)
        self.duration += 1

    @property
    def average_speed(self):
        return np.average(self.speed)

    def get_cumulative_lane_time(self):
        return np.sum(self.lane_time)

    def get_cumulative_distance(self):
        return np.sum(self.travel_distance)

    @property
    def crashed(self) -> int:
        return 1 if self.collision[-1] else 0


# Performance class
class Performance:

    def __init__(self):
        self.average_speed = []
        self.collision = []
        self.travel_distance = []
        self.run_time = []
        self.measurements = 0

    def clear_measurements(self):
        self.__init__()

    def add_measurement(self, log: Logger):
        self.average_speed.append(log.average_speed)
        self.collision.append(log.crashed)
        self.run_time.append(log.duration)
        self.travel_distance.append(log.get_cumulative_distance())
        self.measurements += 1

    def get_indicators(self):
        statistics = {
            'measurements': self.measurements,
            'avg_speeds': self.average_speed,
            'mileage': self.travel_distance,
            'run_times': self.run_time,
            'collisions': self.collision,
        }
        return statistics

    def print_performance(self):
        n = self.measurements
        print('The average speed of', n, 'measurements is:', np.average(self.average_speed))
        print('The average total distance of', n, 'measurements is:', np.average(self.travel_distance))
        print('The average duration time is of', n, 'measurements is:', np.average(self.run_time))
        print('The collision rate of', n, 'measurements is:', np.average(self.collision))

    def string_rep(self):
        n = self.measurements
        return f" The average speed of {n} measurements is: {np.average(self.average_speed)} \n" \
               f" The average total distance of {n} measurements is: {np.average(self.travel_distance)} \n" \
               f" The average duration time is of {n} measurements is: {np.average(self.run_time)} \n" \
               f" The collision rate of {n} measurements is: {np.average(self.collision)} \n" \

    def array_rep(self):
        n = self.measurements
        return [np.average(self.average_speed),
                np.average(self.travel_distance), 
                np.average(self.run_time), np.average(self.collision)]


##############################################################
#                      HELPER FUNCTIONS                      #
##############################################################

# Patching function
def monkey_patcher(old, new, type:str="class"):
    """
    Function to dynamically change classes, objects or modules during runtime globally without modifying their source code.
    
    :param old: Original class|object|module to be patched
    :param new: Custom class|object|module
    :return: Patched version of class|object|module
    """
    available_types = ["class"]
    if type not in available_types:
        print("Patching of this type is not implemented (as of this moment)")
        return old
    else:
        return new


# Environments register function
def register_cumstom_envs():
    """
    register
    """
    gym.register(
        id='merge-in-v0',
        entry_point='functions.merge_in_env:MergeinEnv',
    )

    # Merge in Reward
    gym.register(
        id='merge-in-v3',
        entry_point='functions.merge_in_env:MergeinReward',
    )

    # Merge in with extra vehicles
    gym.register(
        id='merge-in-v4',
        entry_point='functions.merge_in_env:MergeinEnvExtraVehicles',
    )

    # Merge in with an extra lane
    gym.register(
        id='merge-in-v5',
        entry_point='functions.merge_in_env:MergeinEnvExtraLane',
    )


# Sequential directory generator function
def sequential_dir(root:str, return_path:bool=False):
    """
    Create folder for next numbered run.
    """
    latest_num = 0
    
    try:
        dirs = os.listdir(root)
        for d in dirs:
            try:
                latest_num = max(latest_num, int(d.split("_")[1]))
            except ValueError:
                continue
    except OSError:
        pass
    
    path = root + "/run_" + str(latest_num+1)
    os.makedirs(path)
    
    if return_path:
        return path


# Round to multiple function
def round_to_mult(num: int, mult: int):
    """
    Round a number to the nearest multiple of a second number
    """
    return mult * round(num / mult)


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
                     learning_rate=0.0005565336168379374,
                     batch_size=8,
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
                #print("TTC under 20: ", info['TTC'])
            

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





highway_env.vehicle.kinematics.Vehicle = monkey_patcher(highway_env.vehicle.kinematics.Vehicle, MyVehicle)

register_cumstom_envs()
# print("registered")


