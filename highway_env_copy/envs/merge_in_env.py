from typing import Dict, Text

import numpy as np

from highway_env_copy import utils
from highway_env_copy.envs.common.abstract import AbstractEnv
from highway_env_copy.road.lane import LineType, StraightLane, SineLane
from highway_env_copy.road.road import Road, RoadNetwork
from highway_env_copy.vehicle.controller import ControlledVehicle
from highway_env_copy.vehicle.objects import Obstacle
# added
from highway_env.envs.common.observation import TimeToCollisionObservation, KinematicObservation

class MergeinEnv(AbstractEnv):

    """
    A highway merge negotiation environment.
    The ego-vehicle is driving on a highway and approached a merge, with some vehicles incoming on the access ramp.
    It is rewarded for maintaining a high speed and avoiding collisions, but also making room for merging
    vehicles.
    """

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "collision_reward": -1,
            "right_lane_reward": 0.1,
            "high_speed_reward": 0.2,
            "reward_speed_range": [20, 30],
            "merging_speed_reward": -0.5,
            "lane_change_reward": -0.05,
            "other_vehicles": 9
        })
        return cfg

    def _reward(self, action: int) -> float:
        """
        The vehicle is rewarded for driving with high speed on lanes to the right and avoiding collisions

        But an additional altruistic penalty is also suffered if any vehicle on the merging lane has a low speed.

        :param action: the action performed
        :return: the reward of the state-action transition
        """
        reward = sum(self.config.get(name, 0) * reward for name, reward in self._rewards(action).items())
        return utils.lmap(reward,
                          [self.config["collision_reward"] + self.config["merging_speed_reward"],
                           self.config["high_speed_reward"] + self.config["right_lane_reward"]],
                          [0, 1])

    def _rewards(self, action: int) -> Dict[Text, float]:
        #print("action of reward:", action)
        ttc_reward = self._compute_ttc()
        scaled_speed = utils.lmap(self.vehicle.speed, self.config["reward_speed_range"], [0, 1])
        return {
            "collision_reward": self.vehicle.crashed,
            "right_lane_reward": self.vehicle.lane_index[2] / 1,
            "high_speed_reward": scaled_speed,
            #"lane_change_reward": action in [0, 2],
            "merging_speed_reward": sum(  # Altruistic penalty
                (vehicle.target_speed - vehicle.speed) / vehicle.target_speed
                for vehicle in self.road.vehicles
                if vehicle.lane_index == ("b", "c", 2) and isinstance(vehicle, ControlledVehicle)
            )
        }
   
    def _compute_ttc(self):
            TTC = None
            obs_matrix = KinematicObservation(self, absolute=False, vehicles_count=self.config["other_vehicles"],
                                            normalize=False).observe()
            use_TTC = False
            self.glob_TTC = float('inf')
            for vehicle in range(1, len(obs_matrix)):
                x_pos = obs_matrix[vehicle][1]
                y_pos = -1 * obs_matrix[vehicle][2]
                pos_vec = [x_pos, y_pos]  # this is relative when absolute = False
                vx = obs_matrix[vehicle][3]
                vy = -1 * obs_matrix[vehicle][4]
                vel_vec = [vx, vy]
                if np.dot(pos_vec, pos_vec) != 0:
                    proj_pos_vel = np.multiply(np.dot(vel_vec, pos_vec) / np.dot(pos_vec, pos_vec), pos_vec)
                    len_pos = np.linalg.norm(pos_vec)
                    len_proj = np.linalg.norm(proj_pos_vel)

                    if proj_pos_vel[0] * vel_vec[0] > 0 and proj_pos_vel[1] * vel_vec[1] > 0:  # collinear so TTC infinite
                        TTC = float('Inf')
                    else:
                        TTC = len_pos / len_proj
                       
                else:
                    TTC = float('Inf')
                if TTC > 0:  # just to be safe
                    self.glob_TTC = min(self.glob_TTC, TTC)

            if self.glob_TTC < 2:  # only care about TTC if crash is close
                ttc_reward = 1 - 2 / self.glob_TTC
            else:
                ttc_reward = 0

            return ttc_reward

    
    def _info(self, obs, action) -> dict:
        """
        Return a dictionary of additional information

        :param obs: current observation
        :param action: current action
        :return: info dict
        """
        info = {
            "speed": self.vehicle.speed,
            "crashed": self.vehicle.crashed,
            "action": action,
        }
        try:
            info["rewards"] = self._rewards(action)
            info["TTC"] = self.glob_TTC
        except NotImplementedError:
            pass
        return info
    
    def _is_terminated(self) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        # print("crash" + str(self.vehicle.crashed))
        # print("over"  + str(self.vehicle.position[0] > 370))
        return self.vehicle.crashed or bool(self.vehicle.position[0] > 370)

    def _is_truncated(self) -> bool:
        return False

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()

    def _make_road(self) -> None:
        """
        Make a road composed of a straight highway and a merging lane.

        :return: the road
        """
        net = RoadNetwork()

        # Highway lanes
        ends = [150, 80, 80, 150]  # Before, converging, merge, after
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        y = [0, StraightLane.DEFAULT_WIDTH,2*StraightLane.DEFAULT_WIDTH]
        line_type = [[c, s],[n, s], [n, c]]
        line_type_merge = [[c, s],[n, s], [n, s]]
        for i in range(3):
            net.add_lane("a", "b", StraightLane([0, y[i]], [sum(ends[:2]), y[i]], line_types=line_type[i]))
            net.add_lane("b", "c", StraightLane([sum(ends[:2]), y[i]], [sum(ends[:3]), y[i]], line_types=line_type_merge[i]))
            net.add_lane("c", "d", StraightLane([sum(ends[:3]), y[i]], [sum(ends), y[i]], line_types=line_type[i]))

        # Merging lane
        amplitude = 3.25
        ljk = StraightLane([0, 6.5 + 4 + 4+ 4], [ends[0], 6.5 + 4 + 4+ 4], line_types=[c, c], forbidden=True)
        lkb = SineLane(ljk.position(ends[0], -amplitude), ljk.position(sum(ends[:2]), -amplitude),
                       amplitude, 2 * np.pi / (2*ends[1]), np.pi / 2, line_types=[c, c], forbidden=True)
        lbc = StraightLane(lkb.position(ends[1], 0), lkb.position(ends[1], 0) + [ends[2], 0],
                           line_types=[n, c])
        net.add_lane("j", "k", ljk)
        net.add_lane("k", "b", lkb)
        net.add_lane("b", "c", lbc)
        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        road.objects.append(Obstacle(road, lbc.position(ends[2], 0)))
        self.road = road

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.

        :return: the ego-vehicle
        """
        road = self.road
        ego_vehicle = self.action_type.vehicle_class(road,
                                                     road.network.get_lane(("j", "k", 0)).position(30, 0),
                                                     speed=25)
        road.vehicles.append(ego_vehicle)

        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])


        # Generate a list of 20 tuples with random position and speed values
        number_of_cars = 10
        random_cars = [(self.np_random.uniform(0, 1), self.np_random.uniform(27, 30)) for _ in range(number_of_cars)]


        for i , (position, speed) in enumerate(random_cars):
            lane = road.network.get_lane(("a", "b", self.np_random.integers(2)))
            position = lane.position(position + (380/number_of_cars)* i , 0)#+ self.np_random.uniform(-5, 5)
            speed = speed #+= self.np_random.uniform(-1, 1)
            road.vehicles.append(other_vehicles_type(road, position, speed=speed))
        
        # Add a vehicle before and after the ego vehicle on the merging lane

        # merge_lane = road.network.get_lane(("b", "c", 0))
        # road.vehicles.append(other_vehicles_type(road, merge_lane.position(10, 0), 25))
        # road.vehicles.append(other_vehicles_type(road, merge_lane.position(100, 0), 25))

        # merging_v = other_vehicles_type(road, road.network.get_lane(("j", "k", 0)).position(110, 0), speed=20)
        ego_vehicle.target_speed = 30
        # road.vehicles.append(merging_v)
        self.vehicle = ego_vehicle


#DiscreteMetaAction rewards
class MergeinReward(MergeinEnv):

    #normal high speed reward
    def _compute_high_speed_reward(self) -> float:
        speed_range = self.config["reward_speed_range"]
        scaled_speed = utils.lmap(self.vehicle.speed, speed_range, [0, 1])
        return scaled_speed
    
    def _rewards(self, action: int) -> float:
        ttc_reward = self._compute_ttc()
        return 0
    
    def _reward(self, action):
        reward = 0.0

        reward += self._compute_ttc() * 0.5
        #print(f"TTC penalty = {self._compute_ttc()} and reward = {reward}")

        reward += self._compute_high_speed_reward() * 8
        #print(f"compute high speed = {self._compute_high_speed_reward()} and reward = {reward}")

        #    print(f"lane change applied and reward = {reward}")
        if self.vehicle.lane_index[1] == "c":
            reward += 2
            #print(f"right lane applied and reward is {reward}")

        # Conform'd
        reward -= 0.5 * (0.2 * abs(self.vehicle.action["acceleration"]) + 
                                4 / np.pi * abs(self.vehicle.action["steering"]) + 
                                1.0 * abs(self.vehicle.jerk))
        

        if self._is_terminated() and not self.vehicle.crashed:
            reward += 15
        #    print("Car finished!")

        # Scaling of reward
        reward = reward / 25

        if self.vehicle.crashed:
        #    print("car crashed")
            return -20
        
        #print("Reward in merge_in" ,reward)
        return reward

        
# Extra vehicles
class MergeinEnvExtraVehicles(MergeinEnv):
    def __init__(self, config=None, render_mode=None):
        # print("Initializing MergeinEnvSalih")
        super().__init__(config)
        self.render_mode = render_mode

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
                "collision_penalty": -1, # High collision penalty
                "right_lane_reward": 0.3, # Reward for driving in the rightmost lane
                "high_speed_reward": 0.5, # Reward for achieving high speeds within the speed range
                "reward_speed_range": [20, 30], # The speed range at which the agent gets rewarded
                "lane_change_penalty": -0.4, # Penalty for changing lanes
                "ttc_reward_weight": 1, # The weight for the TTC reward
                "other_vehicles": 9 # The number of other vehicles
        })
        return cfg

    def _reward(self,action):
        """
        Set the rewards, sum them and map them to a value between 0 and 1
        """

        rewards = self._rewards(action)
        ttc_reward = rewards["ttc_reward"]
        collision_penalty = rewards["collision_penalty"]
        lane_change_penalty = rewards["lane_change_penalty"]
        high_speed_reward = rewards["high_speed_reward"]
        right_lane_reward = rewards["right_lane_reward"]
        
        reward = (
            ttc_reward
            + collision_penalty
            + lane_change_penalty
            + high_speed_reward
            + right_lane_reward
        )
        reward = sum(self.config.get(name, 0) * reward for name, reward in self._rewards(action).items()) 
        if self._is_terminated() and not self.vehicle.crashed:
            reward += 1

        return utils.lmap(reward,
                          [self.config["collision_penalty"] + self.config["lane_change_penalty"],
                           self.config["high_speed_reward"] + self.config["right_lane_reward"] + self.config["ttc_reward_weight"]],
                          [0, 1])
    
    def _rewards(self, action):
        """
        Code for defining how the rewards are calculated
        """
        #TTC Reward
        ttc_reward = self._compute_ttc()

        #Lane changing penalty only activates after merging
        highway_lanes = ["a", "b", "c"]
        lane_change = action in [0, 2]
        on_highway = self.vehicle.lane_index[1] in highway_lanes
        was_on_highway = self.vehicle.previous_lane_index[1] in highway_lanes if self.vehicle.previous_lane_index else False
        lane_change_penalty = 0
        if lane_change and was_on_highway and on_highway:
            lane_change_penalty = self.config["lane_change_penalty"]
        
        return {
            "ttc_reward": self.config["ttc_reward_weight"] * ttc_reward, 
            "collision_penalty": self.config["collision_penalty"] if self.vehicle.crashed else 0, # Penalty when changing lanes
            "lane_change_penalty": lane_change_penalty ,  # Penalty for changing lanes
            "high_speed_reward": self._compute_high_speed_reward(),
            "right_lane_reward": self.config["right_lane_reward"] if self.vehicle.lane_index[1] == "c" else 0, # Reward for being in the rightmost lane
        }

    def _compute_high_speed_reward(self) -> float:
        speed_range = self.config["reward_speed_range"]
        scaled_speed = utils.lmap(self.vehicle.speed, speed_range, [0, 1])
        return scaled_speed
    
    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()

    
    def _make_road(self) -> None:
        """
        Make a road composed of a straight highway and a merging lane.

        :return: the road
        """
        net = RoadNetwork()

        # Highway lanes
        ends = [150, 80, 80, 150]  # Before, converging, merge, after
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        y = [0, StraightLane.DEFAULT_WIDTH, 2 * StraightLane.DEFAULT_WIDTH]
        line_type = [[c, s], [n, s], [n, s], [n, s]]
        line_type_merge = [[c, s], [n, s], [n, s]]

        # Define 4 highway lanes (including an additional left lane)
        for i in range(3):
            net.add_lane("a", "b", StraightLane([0, y[i]], [sum(ends[:2]), y[i]], line_types=line_type[i]))
            net.add_lane("b", "c", StraightLane([sum(ends[:2]), y[i]], [sum(ends[:3]), y[i]], line_types=line_type[i]))
            net.add_lane("c", "d", StraightLane([sum(ends[:3]), y[i]], [sum(ends), y[i]], line_types=line_type[i]))
    	
        
        
        # Merging lane attached to the rightmost lane "d"
        amplitude = 3.25
        ljk = StraightLane([0, 6.5 + 4 + 4 + 4], [ends[0], 6.5 + 4 + 4 + 4], line_types=[c, c], forbidden=True)
        lkb = SineLane(ljk.position(ends[0], -amplitude), ljk.position(sum(ends[:2]), -amplitude),
                    amplitude, 2 * np.pi / (2 * ends[1]), np.pi / 2, line_types=[c, c], forbidden=True)
        lbc = StraightLane(lkb.position(ends[1], 0), lkb.position(ends[1], 0) + [ends[2], 0],
                        line_types=[n, c])
        net.add_lane("j", "k", ljk)
        net.add_lane("k", "d", lkb)
        net.add_lane("d", "c", lbc)
        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        road.objects.append(Obstacle(road, lbc.position(ends[2], 0)))
        self.road = road


    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.

        :return: the ego-vehicle
        """
        road = self.road
        ego_vehicle = self.action_type.vehicle_class(road,
                                                     road.network.get_lane(("j", "k", 0)).position(30, 0),
                                                     speed=25)
        road.vehicles.append(ego_vehicle)

        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])


        
        number_of_cars = 15
        random_cars = [(self.np_random.uniform(0, 1), self.np_random.uniform(27, 30)) for _ in range(number_of_cars)]


        for i , (position, speed) in enumerate(random_cars):
            lane = road.network.get_lane(("a", "b", self.np_random.integers(2)))
            position = lane.position(position + (380/number_of_cars)* i , 0)
            speed = speed 
            road.vehicles.append(other_vehicles_type(road, position, speed=speed))
    
        ego_vehicle.target_speed = 30
        self.vehicle = ego_vehicle

class MergeinEnvExtraLane(MergeinEnv):
    def __init__(self, config=None, render_mode=None):
        # print("Initializing MergeinEnvSalih")
        super().__init__(config)
        self.render_mode = render_mode

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
                "collision_penalty": -1, # changed from -1
                "right_lane_reward": 0.3,
                "high_speed_reward": 0.5, #Look at which value the car also brakes instead of only overtaking
                "reward_speed_range": [20, 30], #speed range can differ
                "lane_change_penalty": -0.01, #mogelijk hoger, nog kijken voor merging
                "ttc_reward_weight": 1,
                "other_vehicles": 9
        })
        return cfg

    def _reward(self,action):
        rewards = self._rewards(action)

        ttc_reward = rewards["ttc_reward"]
        collision_penalty = rewards["collision_penalty"]
        lane_change_penalty = rewards["lane_change_penalty"]
        high_speed_reward = rewards["high_speed_reward"]
        right_lane_reward = rewards["right_lane_reward"]
        
        reward = (
            ttc_reward
            + collision_penalty
            + lane_change_penalty
            + high_speed_reward
            + right_lane_reward
        )
        reward = sum(self.config.get(name, 0) * reward for name, reward in self._rewards(action).items()) 
        if self._is_terminated() and not self.vehicle.crashed:
            reward += 1

       
        return utils.lmap(reward,
                          [self.config["collision_penalty"] + self.config["lane_change_penalty"],
                           self.config["high_speed_reward"] + self.config["right_lane_reward"] + self.config["ttc_reward_weight"]],
                          [0, 1])
    


    def _rewards(self, action):
        ttc_reward = self._compute_ttc()
        #self.config["lane_change_penalty"] if action in [0, 2] else 0
        return {
            "ttc_reward": self.config["ttc_reward_weight"] * ttc_reward,
            "collision_penalty": self.config["collision_penalty"] if self.vehicle.crashed else 0,
            "lane_change_penalty": self.config["lane_change_penalty"] ,  # Penalty for changing lanes
            "high_speed_reward": self._compute_high_speed_reward(),
            "right_lane_reward": self.config["right_lane_reward"] if self.vehicle.lane_index[1] == "d" else 0, # Reward for being in the rightmost lane  
        }


     #normal high speed reward
    def _compute_high_speed_reward(self) -> float:
        speed_range = self.config["reward_speed_range"]
        scaled_speed = utils.lmap(self.vehicle.speed, speed_range, [0, 1])
        return scaled_speed

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()
        
    
    def _make_road(self) -> None:
        """
        Make a road composed of a straight highway and a merging lane.

        :return: the road
        """
        net = RoadNetwork()

        # Highway lanes
        ends = [150, 80, 80, 150]  # Before, converging, merge, after
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        y = [0, StraightLane.DEFAULT_WIDTH,2*StraightLane.DEFAULT_WIDTH, 3*StraightLane.DEFAULT_WIDTH]
        line_type = [[c, s],[n, s], [n, s], [n, c]]
        line_type_merge = [[c, s],[n, s], [n, s], [n, s]]

        for i in range(4):
            net.add_lane("a", "b", StraightLane([0, y[i]], [sum(ends[:2]), y[i]], line_types=line_type[i]))
            net.add_lane("b", "c", StraightLane([sum(ends[:2]), y[i]], [sum(ends[:3]), y[i]], line_types=line_type_merge[i]))
            net.add_lane("c", "d", StraightLane([sum(ends[:3]), y[i]], [sum(ends), y[i]], line_types=line_type[i]))

        # Merging lane attached to the rightmost lane "d"
        amplitude = 3.25
        ljk = StraightLane([0, 6.5 + 4 + 4 + 4+4], [ends[0], 6.5 + 4 + 4 + 4+4], line_types=[c, c], forbidden=True)
        lkb = SineLane(ljk.position(ends[0], -amplitude), ljk.position(sum(ends[:2]), -amplitude),
                    amplitude, 2 * np.pi / (2 * ends[1]), np.pi / 2, line_types=[c, c], forbidden=True)
        lbc = StraightLane(lkb.position(ends[1], 0), lkb.position(ends[1], 0) + [ends[2], 0],
                        line_types=[n, c])
        net.add_lane("j", "k", ljk)
        net.add_lane("k", "b", lkb)
        net.add_lane("b", "c", lbc)
        
        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        road.objects.append(Obstacle(road, lbc.position(ends[2], 0)))
        self.road = road

class MergeinEnvRandom(MergeinEnv):
    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.

        :return: the ego-vehicle
        """
        road = self.road
        ego_vehicle = self.action_type.vehicle_class(road,
                                                     road.network.get_lane(("j", "k", 0)).position(30, 0),
                                                     speed=25)
        road.vehicles.append(ego_vehicle)

        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])


        # Generate a list of 20 tuples with random position and speed values
        number_of_cars = 10
        random_cars = [(self.np_random.uniform(0, 1), self.np_random.uniform(27, 30)) for _ in range(number_of_cars)]


        for i , (position, speed) in enumerate(random_cars):
            lane = road.network.get_lane(("a", "b", self.np_random.integers(2)))
            position = lane.position(position + (380/number_of_cars)* i , 0) + self.np_random.uniform(-5, 5)
            speed += self.np_random.uniform(-1, 1)
            road.vehicles.append(other_vehicles_type(road, position, speed=speed))
        
        # Add a vehicle before and after the ego vehicle on the merging lane

        # merge_lane = road.network.get_lane(("b", "c", 0))
        # road.vehicles.append(other_vehicles_type(road, merge_lane.position(10, 0), 25))
        # road.vehicles.append(other_vehicles_type(road, merge_lane.position(100, 0), 25))

        # merging_v = other_vehicles_type(road, road.network.get_lane(("j", "k", 0)).position(110, 0), speed=20)
        ego_vehicle.target_speed = 30
        # road.vehicles.append(merging_v)
        self.vehicle = ego_vehicle