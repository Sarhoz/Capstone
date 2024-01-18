from typing import Union, Optional, Tuple, List
import numpy as np
import copy
from collections import deque

from highway_env_copy import utils
from highway_env_copy.road.road import Road, LaneIndex
from highway_env_copy.vehicle.objects import RoadObject, Obstacle, Landmark
from highway_env_copy.utils import Vector

#added
# from highway_env_copy.envs.common.observation import KinematicObservation


class Vehicle(RoadObject):

    """
    A moving vehicle on a road, and its kinematics.

    The vehicle is represented by a dynamical system: a modified bicycle model.
    It's state is propagated depending on its steering and acceleration actions.
    """

    LENGTH = 5.0
    """ Vehicle length [m] """
    WIDTH = 2.0
    """ Vehicle width [m] """
    DEFAULT_INITIAL_SPEEDS = [23, 25]
    """ Range for random initial speeds [m/s] """
    MAX_SPEED = 40.
    """ Maximum reachable speed [m/s] """
    MIN_SPEED = -40.
    """ Minimum reachable speed [m/s] """
    HISTORY_SIZE = 30
    """ Length of the vehicle state history, for trajectory display"""
    # --- Group ---
    COMFORT_ACC_MAX = 3.0  # [m/s2]
    """Desired maximum acceleration."""
    COMFORT_ACC_MIN = -5.0  # [m/s2]
    """Desired maximum deceleration."""
    # --- Group ---

    def __init__(self,
                 road: Road,
                 position: Vector,
                 heading: float = 0,
                 speed: float = 0,
                 predition_type: str = 'constant_steering'):
        super().__init__(road, position, heading, speed)
        self.prediction_type = predition_type
        self.action = {'steering': 0, 'acceleration': 0}
        #--- Group ---
        self.recorded_actions = [{'steering': 0, 'acceleration': 0}]
        self.recorded_positions = [position.tolist()]
        self.destination_location = np.array([0,0])
        self.track_affiliated_lane = False   
        #---Group ---
        self.crashed = False
        self.impact = None
        self.log = []
        self.history = deque(maxlen=self.HISTORY_SIZE)

    @classmethod
    def create_random(cls, road: Road,
                      speed: float = None,
                      lane_from: Optional[str] = None,
                      lane_to: Optional[str] = None,
                      lane_id: Optional[int] = None,
                      spacing: float = 1) \
            -> "Vehicle":
        """
        Create a random vehicle on the road.

        The lane and /or speed are chosen randomly, while longitudinal position is chosen behind the last
        vehicle in the road with density based on the number of lanes.

        :param road: the road where the vehicle is driving
        :param speed: initial speed in [m/s]. If None, will be chosen randomly
        :param lane_from: start node of the lane to spawn in
        :param lane_to: end node of the lane to spawn in
        :param lane_id: id of the lane to spawn in
        :param spacing: ratio of spacing to the front vehicle, 1 being the default
        :return: A vehicle with random position and/or speed
        """
        _from = lane_from or road.np_random.choice(list(road.network.graph.keys()))
        _to = lane_to or road.np_random.choice(list(road.network.graph[_from].keys()))
        _id = lane_id if lane_id is not None else road.np_random.choice(len(road.network.graph[_from][_to]))
        lane = road.network.get_lane((_from, _to, _id))
        if speed is None:
            if lane.speed_limit is not None:
                speed = road.np_random.uniform(0.7*lane.speed_limit, 0.8*lane.speed_limit)
            else:
                speed = road.np_random.uniform(Vehicle.DEFAULT_INITIAL_SPEEDS[0], Vehicle.DEFAULT_INITIAL_SPEEDS[1])
        default_spacing = 12+1.0*speed
        offset = spacing * default_spacing * np.exp(-5 / 40 * len(road.network.graph[_from][_to]))
        x0 = np.max([lane.local_coordinates(v.position)[0] for v in road.vehicles]) \
            if len(road.vehicles) else 3*offset
        x0 += offset * road.np_random.uniform(0.9, 1.1)
        v = cls(road, lane.position(x0, 0), lane.heading_at(x0), speed)
        return v

    @classmethod
    def create_from(cls, vehicle: "Vehicle") -> "Vehicle":
        """
        Create a new vehicle from an existing one.

        Only the vehicle dynamics are copied, other properties are default.

        :param vehicle: a vehicle
        :return: a new vehicle at the same dynamical state
        """
        v = cls(vehicle.road, vehicle.position, vehicle.heading, vehicle.speed)
        if hasattr(vehicle, 'color'):
            v.color = vehicle.color
        return v

    def act(self, action: Union[dict, str] = None) -> None:
        """
        Store an action to be repeated.

        :param action: the input action
        """
        if action:
            self.action = action

    def step(self, dt: float) -> None:
        """
        Propagate the vehicle state given its actions.

        Integrate a modified bicycle model with a 1st-order response on the steering wheel dynamics.
        If the vehicle is crashed, the actions are overridden with erratic steering and braking until complete stop.
        The vehicle's current lane is updated.

        :param dt: timestep of integration of the model [s]
        """
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
        # --- Group ---
        self.recorded_actions.append(self.action)
        self.recorded_positions.append(self.position.tolist())
        # -- Group ---
        
        #--- Group ---
    @property
    def lane_distance(self) -> float:
        return self.lane.distance(self.position)
    
    #Signed lane heading difference. Wrapping perserves the sign.
    #Remark: The sign of the multiplication of lane_distance and lane_difference_heading is
    # - Positive, whenever the car if deviating from the road
    # - Negative, whenever the car is heading to the road
    @property
    def lane_heading_difference(self) -> float:
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

    # jerk only used for continuous!
    @property
    def jerk(self) -> float:
        if len(self.recorded_actions) < 2:
            return 0
        jerk_accel = abs(self.recorded_actions[-2]['acceleration'] - self.recorded_actions[-1]['acceleration']) / (
                self.COMFORT_ACC_MAX - self.COMFORT_ACC_MIN)
        jerk_steer = abs(self.recorded_actions[-2]['steering'] - self.recorded_actions[-1]['steering']) * 2 / np.pi
        return (jerk_accel + jerk_steer) / 2
    
    def clip_actions(self) -> None:
        if self.crashed:
            self.action['steering'] = 0
            self.action['acceleration'] = -1.0*self.speed
        self.action['steering'] = float(self.action['steering'])
        self.action['acceleration'] = float(self.action['acceleration'])
        if self.speed > self.MAX_SPEED:
            self.action['acceleration'] = min(self.action['acceleration'], 1.0 * (self.MAX_SPEED - self.speed))
        elif self.speed < self.MIN_SPEED:
            self.action['acceleration'] = max(self.action['acceleration'], 1.0 * (self.MIN_SPEED - self.speed))

    def on_state_update(self) -> None:
        if self.road:
            self.lane_index = self.road.network.get_closest_lane_index(self.position, self.heading)
            self.lane = self.road.network.get_lane(self.lane_index)
            if self.road.record_history:
                self.history.appendleft(self.create_from(self))

    def predict_trajectory_constant_speed(self, times: np.ndarray) -> Tuple[List[np.ndarray], List[float]]:
        if self.prediction_type == 'zero_steering':
            action = {'acceleration': 0.0, 'steering': 0.0}
        elif self.prediction_type == 'constant_steering':
            action = {'acceleration': 0.0, 'steering': self.action['steering']}
        else:
            raise ValueError("Unknown predition type")

        dt = np.diff(np.concatenate(([0.0], times)))

        positions = []
        headings = []
        v = copy.deepcopy(self)
        v.act(action)
        for t in dt:
            v.step(t)
            positions.append(v.position.copy())
            headings.append(v.heading)
        return (positions, headings)

    @property
    def velocity(self) -> np.ndarray:
        return self.speed * self.direction  # TODO: slip angle beta should be used here

    @property
    def destination(self) -> np.ndarray:
        if getattr(self, "route", None):
            last_lane_index = self.route[-1]
            last_lane_index = last_lane_index if last_lane_index[-1] is not None else (*last_lane_index[:-1], 0)
            last_lane = self.road.network.get_lane(last_lane_index)
            return last_lane.position(last_lane.length, 0)
        else:
            return self.position

    # --- Group ---
    @property
    def destination_key(self) -> np.ndarray:
        if getattr(self, "route", None):
            last_lane_index = self.route[-1]
            last_lane_index = last_lane_index if last_lane_index[-1] is not None else (*last_lane_index[:-1], 0)
            return last_lane_index
        else:
            return self.position
    # --- Group ---


    @property
    def destination_direction(self) -> np.ndarray:
        if (self.destination != self.position).any():
            return (self.destination - self.position) / np.linalg.norm(self.destination - self.position)
        else:
            return np.zeros((2,))

    @property
    def lane_offset(self) -> np.ndarray:
        if self.lane is not None:
            long, lat = self.lane.local_coordinates(self.position)
            ang = self.lane.local_angle(self.heading, long)
            return np.array([long, lat, ang])
        else:
            return np.zeros((3,))
    
    # Added TTC
    # @property
    # def _compute_ttc(self):
    #     TTC = None
    #     from highway_env.envs.common.observation import TimeToCollisionObservation, KinematicObservation
    #     obs_matrix = KinematicObservation(self, absolute=False, vehicles_count=10,
    #                                      normalize=False).observe()
    #     use_TTC = False
    #     glob_TTC = float('inf')
    #     for vehicle in range(1, len(obs_matrix)):
    #         x_pos = obs_matrix[vehicle][1]
    #         y_pos = -1 * obs_matrix[vehicle][2]
    #         pos_vec = [x_pos, y_pos]  # this is relative when absolute = False
    #         vx = obs_matrix[vehicle][3]
    #         vy = -1 * obs_matrix[vehicle][4]
    #         vel_vec = [vx, vy]
    #         if np.dot(pos_vec, pos_vec) != 0:
    #             proj_pos_vel = np.multiply(np.dot(vel_vec, pos_vec) / np.dot(pos_vec, pos_vec), pos_vec)
    #             len_pos = np.linalg.norm(pos_vec)
    #             len_proj = np.linalg.norm(proj_pos_vel)

    #             if proj_pos_vel[0] * vel_vec[0] > 0 and proj_pos_vel[1] * vel_vec[1] > 0:  # collinear so TTC infinite
    #                 TTC = float('Inf')
    #             else:
    #                 TTC = len_pos / len_proj
    #         else:
    #             TTC = float('Inf')
    #         return TTC
        
    # End of added TTC
          

    def to_dict(self, origin_vehicle: "Vehicle" = None, observe_intentions: bool = True) -> dict:
        d = {
            'presence': 1,
            'x': self.position[0],
            'y': self.position[1],
            'vx': self.velocity[0],
            'vy': self.velocity[1],
            'heading': self.heading,
            'cos_h': self.direction[0],
            'sin_h': self.direction[1],
            'cos_d': self.destination_direction[0],
            'sin_d': self.destination_direction[1],
            'long_off': self.lane_offset[0],
            'lat_off': self.lane_offset[1],
            'ang_off': self.lane_offset[2],
        }
        if not observe_intentions:
            d["cos_d"] = d["sin_d"] = 0
        if origin_vehicle:
            origin_dict = origin_vehicle.to_dict()
            for key in ['x', 'y', 'vx', 'vy']:
                d[key] -= origin_dict[key]
        return d

    def __str__(self):
        return "{} #{}: {}".format(self.__class__.__name__, id(self) % 1000, self.position)

    def __repr__(self):
        return self.__str__()
    

    # --- EVERTTHING BELOW IS FROM THE GROUP ---

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
            v.position_change)  # TODO modify to proper distance based on lane progression, in stead of car travel distance
        self.duration += 1
        # self.ttc.append(v._compute_ttc)
        # self.right_lane.append(v.)

    @property
    def average_speed(self):
        return np.average(self.speed)
    
    # @property
    # def minimum_ttc(self):
    #     return np.min(self.ttc)

    # @property
    # def average_ttc(self):
    #     return np.average(self.ttc)


    def get_cumulative_lane_time(self):
        return np.sum(self.lane_time)

    def get_cumulative_distance(self):
        return np.sum(self.travel_distance)

    @property
    def crashed(self) -> int:
        return 1 if self.collision[-1] else 0


class Performance:

    def __init__(self):
        self.average_speed = []
        self.collision = []
        self.travel_distance = []
        self.run_time = []
        self.measurements = 0
        # self.avg_ttc = []
        # self.min_ttc = []

    def clear_measurements(self):
        self.__init__()

    def add_measurement(self, log: Logger):
        self.average_speed.append(log.average_speed)
        self.collision.append(log.crashed)
        self.run_time.append(log.duration)
        self.travel_distance.append(log.get_cumulative_distance())
        self.measurements += 1
        # self.avg_ttc.append(log.average_ttc)
        # self.min_ttc.append(log.minimum_ttc)

    def get_indicators(self):
        statistics = {
            'measurements': self.measurements,
            'avg_speeds': self.average_speed,
            'mileage': self.travel_distance,
            'run_times': self.run_time,
            'collisions': self.collision,
            # 'min_ttc': self.avg_ttc,
            # 'average ttc' : self.avg_ttc,
        }
        return statistics

    def print_performance(self):
        n = self.measurements
        print('The average speed of', n, 'measurements is:', np.average(self.average_speed))
        print('The average total distance of', n, 'measurements is:', np.average(self.travel_distance))
        print('The average duration time is of', n, 'measurements is:', np.average(self.run_time))
        print('The collision rate of', n, 'measurements is:', np.average(self.collision))
        # print('The average minimal ttc of ', n, 'measurements is:', np.average(self.min_ttc))
        # print('The average of the average ttc of ', n, 'measurements is:', np.average(self.min_ttc)) 
        # Voegt niet zoveel toe

    def string_rep(self):
        n = self.measurements
        return f" The average speed of {n} measurements is: {np.average(self.average_speed)} \n" \
               f" The average total distance of {n} measurements is: {np.average(self.travel_distance)} \n" \
               f" The average duration time is of {n} measurements is: {np.average(self.run_time)} \n" \
               f" The collision rate of {n} measurements is: {np.average(self.collision)} \n" \
            #    f" The average minimal ttc of ', {n}, 'measurements is:', {np.average(self.min_ttc)} \n" \

    def array_rep(self):
        n = self.measurements
        return [np.average(self.average_speed),
                np.average(self.travel_distance), 
                np.average(self.run_time), np.average(self.collision)]
