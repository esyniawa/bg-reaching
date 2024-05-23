import matplotlib.pyplot as plt
import numpy as np

from network.params import parameters, state_space
from network.utils import bivariate_gauss, circ_gauss
from network.model import *

from kinematics.planar_arms import PlanarArms


def make_inputs(start_point: list[float, float] | tuple[float, float],
                end_point: list[float, float] | tuple[float, float],
                distance_rate: float = 30.,
                trace_factor: float = .5,
                training_trace: bool = True,
                show_input: bool = False):

    motor_angle, distance = PlanarArms.calc_motor_vector(init_pos=np.array(start_point),
                                                         end_pos=np.array(end_point),
                                                         arm=parameters['moving_arm'])

    # calculate pm input
    input_pm = bivariate_gauss(xy=state_space,
                               mu=end_point,
                               sigma=parameters['sig_pm'],
                               norm=True,
                               plot=show_input)

    if training_trace:
        n_trace = int(np.floor(distance/distance_rate))
        if n_trace > 0:
            trace = np.linspace(start=start_point, stop=end_point, endpoint=False, num=n_trace)
            for point in trace:
                input_pm += trace_factor * bivariate_gauss(xy=state_space,
                                                           mu=point,
                                                           sigma=parameters['sig_pm'],
                                                           norm=True)

    # calculate s1 input
    input_s1 = bivariate_gauss(xy=state_space,
                               mu=start_point,
                               sigma=parameters['sig_s1'],
                               norm=True,
                               plot=show_input)

    # calculate motor input
    input_m1 = circ_gauss(mu=motor_angle, sigma=parameters['sig_m1'],
                          n=parameters['dim_motor'], scal=parameters['motor_step_size'], norm=False)

    return input_pm, input_s1, input_m1, distance


def train_position(init_position: np.ndarray,
                   t_reward: float = 300.,
                   t_wait: float = 50.) -> np.ndarray:

    random_x = np.random.uniform(low=parameters['x_reaching_space_limits'][0],
                                 high=parameters['x_reaching_space_limits'][1])
    random_y = np.random.uniform(low=parameters['y_reaching_space_limits'][0],
                                 high=parameters['y_reaching_space_limits'][1])

    base_pm, base_s1, base_m1, distance = make_inputs(start_point=init_position,
                                                      end_point=[random_x, random_y])

    # simulation state
    PM.baseline = 0
    S1.baseline = 0
    CM.baseline = 0
    ann.simulate(t_wait)

    # set inputs
    S1.baseline = base_s1
    CM.baseline = base_m1
    ann.simulate(100.)

    # send reward
    SNc.firing = 1
    PM.baseline = base_pm
    ann.simulate_until(t_reward, population=SNr)
    SNc.firing = 0
    PM.baseline = 0

    ann.reset(populations=True, monitors=False)

    # return new position
    return np.array([random_x, random_y])


def train_fixed_position(init_position: np.ndarray,
                         goal: np.ndarray,
                         t_reward: float = 300.,
                         t_wait: float = 50.) -> None:

    base_pm, base_s1, base_m1, distance = make_inputs(start_point=init_position,
                                                      end_point=goal)

    # simulation state
    PM.baseline = 0
    S1.baseline = 0
    CM.baseline = 0
    ann.simulate(t_wait)

    # set inputs
    S1.baseline = base_s1
    CM.baseline = base_m1
    ann.simulate(100.)

    # send reward
    SNc.firing = 1
    PM.baseline = base_pm
    ann.simulate_until(t_reward, population=SNr)
    SNc.firing = 0
    PM.baseline = 0

    ann.reset(populations=True, monitors=False)


def test_movement(scale_movement: float = 1.0, t_wait: float = 50.) -> None:

    points_to_follow = [
        np.array((-100, 200)),
        np.array((-100, 50)),
        np.array((100, 50)),
        np.array((100, 200)),
    ]

    n_points = len(points_to_follow)

    # simulate movement
    for i, point in enumerate(points_to_follow):

        # make inputs for PM
        input_pm, input_s1, _, distance = make_inputs(start_point=point,
                                                      end_point=points_to_follow[(i+1) % n_points],
                                                      training_trace=False)

        # simulation state
        SNc.firing = 0
        PM.baseline = 0
        S1.baseline = 0
        CM.baseline = 0
        ann.simulate(t_wait)

        # set inputs
        PM.baseline = input_pm
        S1.baseline = input_s1
        ann.simulate(distance * scale_movement)

        ann.reset(monitors=False)
