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

    print('default')

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

class MergeinEnvArno(MergeinEnv):
    "new merge-in environment made by Arno"

    print('Arno')

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

        })
        return cfg

    def _rewards(self, action: int) -> Dict[Text, float]:
        scaled_speed = utils.lmap(self.vehicle.speed, self.config["reward_speed_range"], [0, 1])
        return {
            "collision_reward": self.vehicle.crashed,
            "right_lane_reward": self.vehicle.lane_index[2] / 1,
            "high_speed_reward": scaled_speed,
            "lane_change_reward": action in [0, 2],
            "merging_speed_reward": sum(  # Altruistic penalty
                (vehicle.target_speed - vehicle.speed) / vehicle.target_speed
                for vehicle in self.road.vehicles
                if vehicle.lane_index == ("b", "c", 2) and isinstance(vehicle, ControlledVehicle)
            )
        }
    
    # Sarhoz
class MergeinEnvSarhoz(MergeinEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
                "observation": {
                "type": "Kinematics"},
                "action": {
                "type": "ContinuousAction",
                "longitudinal": True,
                "lateral": True,
                "target_speeds": [20, 30],
                "other_vehicles": 9
                },
                "other_vehicles": 9,
                "collision_reward": -1,
                "lane_centering_cost": 4,
                "lane_centering_reward": 1})
            
        return config
        
    def _reward(self, action: np.ndarray) -> float:
        reward = 0
        rewards = self._rewards(action)
        reward_weights = {'safety': 0.50, 'comfort': 0.25, 'efficiency': 0.25}
        for key in rewards.keys():
            reward += reward_weights[key] * rewards[key]
        # apply the penalties for not abiding by the rules
        # if self.vehicle.speed > speed_limit:
        #     reward -= (self.vehicle.speed - speed_limit) / (self.vehicle.MAX_SPEED - speed_limit)

        if not self.vehicle.on_road:
            reward -= 10
        if self.vehicle.crashed:
            reward -= 35
        #print('time for reward')
        return reward

        
    def _rewards(self, action: np.ndarray) -> Dict[Text, float]:
        #(2 * self.vehicle.speed) / (
        #        self.vehicle.MAX_SPEED - self.vehicle.MIN_SPEED)  # max speed returns a reward of 1
        TTC = None
        obs_matrix = KinematicObservation(self, absolute=False, vehicles_count=self.config["other_vehicles"], normalize=False).observe()
        use_TTC = False
        glob_TTC = float('inf')
        for vehicle in range(1,len(obs_matrix)):
            x_pos = obs_matrix[vehicle][1]
            y_pos = -1 * obs_matrix[vehicle][2]
            pos_vec = [x_pos, y_pos] # this is relative when absolute = False
            vx = obs_matrix[vehicle][3]
            vy = -1 * obs_matrix[vehicle][4]
            vel_vec = [vx, vy]
            if np.dot(pos_vec, pos_vec) != 0:
                proj_pos_vel = np.multiply(np.dot(vel_vec, pos_vec) / np.dot(pos_vec, pos_vec), pos_vec)
                len_pos = np.linalg.norm(pos_vec)
                len_proj = np.linalg.norm(proj_pos_vel)

                if proj_pos_vel[0] * vel_vec[0] > 0 and proj_pos_vel[1] * vel_vec[1] > 0: # collinear so TTC infinite
                    TTC = float('Inf')
                else: 
                    TTC = len_pos / len_proj  
            else:
                TTC = float('Inf')          
            if TTC > 0: # just to be safe
                glob_TTC = min(glob_TTC, TTC)
        #print(glob_TTC)    
        if glob_TTC < 3: # only care about TTC if crash is close
            use_TTC = True


        _, lateral = self.vehicle.lane.local_coordinates(self.vehicle.position)
        if not use_TTC:   
        # all reward components are normalized (in range 0 to 1) and we take the weighted average of them in _reward
            safety_reward = 1 / (1 + self.config["lane_centering_cost"] * lateral ** 2) 
        else:
            #(1 + self.config["lane_centering_cost"] * lateral ** 2)
            safety_reward = 0.8 * (1 - 2 / TTC) + 0.2 * 1 / (1 + self.config["lane_centering_cost"] * lateral ** 2) 
            #(1 + 4 * self.vehicle.lane.distance(self.vehicle.position) ** 2) 
        speed_limit = 10
        if self.vehicle.speed <= speed_limit:
            efficiency_reward = self.vehicle.speed / speed_limit
        else:
            efficiency_reward = 1 - self.vehicle.speed / (self.vehicle.MAX_SPEED)


        comfort_reward = 1 - self.vehicle.jerk
        rewards_keys = ['safety', 'comfort', 'efficiency']
        rewards_values = [safety_reward, comfort_reward, efficiency_reward]
        return dict(zip(rewards_keys, rewards_values))


#Salih Discrete rewards
class MergeinEnvSalih(MergeinEnv):
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
                "collision_penalty": -1,
                "right_lane_reward": 0.1,
                "high_speed_reward": 0.2,
                "reward_speed_range": [20, 30],
                "merging_speed_penalty": -0.5,
                "lane_change_penalty": -0.05,
                "ttc_reward_weight": 3 # kan wss weg als we niet gebruiken,
        })

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

        if self.is_terminated() and not self.vehicle.crashed:
            reward += 1
        
        return utils.lmap(reward, [0, 1], [self.config["collision_penalty"] + self.config["merging_speed_penalty"], 1])
    
    def _rewards(self, action):
        ttc_reward, ttc = self._compute_ttc()
        if ttc < 3:
            ttc_reward = 1 - 3 / ttc
        else:
            ttc_reward = 0

        return {
            "ttc_reward": self.config["ttc_reward_weight"] * ttc_reward,
            "collision_penalty": self.vehicle.crashed,
            "lane_change_penalty": int(action in [0, 1]),  # Penalty for changing lanes
            "high_speed_reward": self._compute_high_speed_reward(),
            "right_lane_reward": int(self.vehicle.lane_index[2] == 0),  # Reward for being in the rightmost lane
        }
    
    @staticmethod
    def _compute_vehicle_ttc(ego_vehicle, other_vehicle):
        ego_radius = ego_vehicle.LENGTH / 2
        other_radius = other_vehicle.LENGTH / 2 
        ego_position = ego_vehicle.position
        other_position = other_vehicle.position
        relative_position = ego_position - other_position
        relative_speed = ego_vehicle.speed - other_vehicle.speed

        time_to_collision = (
            np.dot(relative_position, relative_speed)
            / np.dot(relative_speed, relative_speed)
        )
        if time_to_collision < 0:
            time_to_collision = float("inf")
        else:
            distance = np.linalg.norm(relative_position - time_to_collision * relative_speed)
            if distance < ego_radius + other_radius:
                return time_to_collision
            else:
                return float("inf")

    def _compute_high_speed_reward(self) -> float:
        speed_range = self.config["reward_speed_range"]
        scaled_speed = utils.lmap(self.vehicle.speed, speed_range, [0, 1])
        return scaled_speed


