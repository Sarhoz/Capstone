"""
Utility file containing helper functions and custom classes.
"""
import numpy as np
import os

from highway_env.road.road import Road
from highway_env.utils import Vector
from highway_env.vehicle.kinematics import Vehicle


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
# TODO add logger cls

# Performance class
# TODO add perf cls






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
# TODO add register func


# Sequential directory generator function
def sequential_dir(root:str, return_path:bool=False):
    dirs = os.listdir(root)
    latest_num = 0
    for d in dirs:
        try:
            latest_num = max(latest_num, int(d.split("_")[1]))
        except ValueError:
            continue
    path = root + "/run_" + str(latest_num+1)
    os.makedirs(path)
    
    if return_path:
        return path







