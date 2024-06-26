import matplotlib.pyplot as plt
import numpy as np

from network.params import parameters, state_space
from network.utils import bivariate_gauss, circ_gauss
from network.model import *

from kinematics.planar_arms import PlanarArms


def generate_random_coordinate(x_bounds_lower: float = parameters['x_reaching_space_limits'][0],
                               x_bounds_upper: float = parameters['x_reaching_space_limits'][1],
                               y_bounds_lower: float = parameters['y_reaching_space_limits'][0],
                               y_bounds_upper: float = parameters['y_reaching_space_limits'][1],
                               clip_border: float = 10.):

    random_x = np.random.uniform(low=x_bounds_lower + clip_border,
                                 high=x_bounds_upper - clip_border)

    random_y = np.random.uniform(low=y_bounds_lower + clip_border,
                                 high=y_bounds_upper - clip_border)

    return [random_x, random_y]


def make_inputs(start_point: list[float, float] | tuple[float, float],
                end_point: list[float, float] | tuple[float, float],
                distance_rate: float = 40.,
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
                               plot=show_input)

    if training_trace:
        n_trace = int(np.floor(distance/distance_rate))
        if n_trace > 0:
            trace = np.linspace(start=start_point, stop=end_point, endpoint=False, num=n_trace)
            for point in trace:
                input_pm += trace_factor * bivariate_gauss(xy=state_space,
                                                           mu=point,
                                                           sigma=parameters['sig_pm'])

    # calculate s1 input
    input_s1 = bivariate_gauss(xy=state_space,
                               mu=start_point,
                               sigma=parameters['sig_s1'],
                               plot=show_input)

    # calculate motor input
    input_m1 = circ_gauss(mu=motor_angle, sigma=parameters['sig_m1'],
                          n=parameters['dim_motor'], scal=parameters['motor_step_size'], norm=False)

    return input_pm, input_s1, input_m1, distance


def train_position(init_position: np.ndarray,
                   t_reward: float = 350.,
                   t_wait: float = 50.,
                   trace: bool = True,
                   return_sim_time: bool = False) -> np.ndarray:

    new_position = generate_random_coordinate()

    # make input
    base_pm, base_s1, base_m1, distance = make_inputs(start_point=init_position,
                                                      end_point=new_position,
                                                      training_trace=trace)

    # simulation state
    SNc.firing = 0
    PM.baseline = 0
    S1.baseline = 0
    CM.baseline = 0
    ann.simulate(t_wait)

    # send reward and set inputs
    SNc.firing = 1
    PM.baseline = base_pm
    S1.baseline = base_s1
    CM.baseline = base_m1
    time = ann.simulate_until(t_reward, population=SNr)

    ann.reset(populations=True, monitors=False)

    # return new position
    if return_sim_time:
        return np.array(new_position), time
    else:
        return np.array(new_position)


def train_fixed_position(init_position: np.ndarray,
                         goal: np.ndarray,
                         t_reward: float = 300.,
                         t_wait: float = 50.,
                         trace: bool = True) -> None:

    base_pm, base_s1, base_m1, distance = make_inputs(start_point=init_position,
                                                      end_point=goal, training_trace=trace)

    # simulation state
    SNc.firing = 0
    PM.baseline = 0
    S1.baseline = 0
    CM.baseline = 0
    ann.simulate(t_wait)

    # send reward and set inputs
    SNc.firing = 1
    PM.baseline = base_pm
    S1.baseline = base_s1
    CM.baseline = base_m1
    ann.simulate_until(t_reward, population=SNr)

    ann.reset(populations=True, monitors=False)


def test_movement(scale_movement: float = 1.0,
                  scale_pm: float = 5.0,
                  scale_s1: float = 5.0,
                  t_wait: float = 50.) -> None:

    # disable learning
    ann.disable_learning()

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
        PM.baseline = scale_pm * input_pm
        S1.baseline = scale_s1 * input_s1
        ann.simulate(distance * scale_movement)

        ann.reset(monitors=False)


def sim_movement_m1_input(sim_time: float = 500.,
                          plasticity_snr: bool = True,
                          plasticity_m1: bool = False,
                          t_wait: float = 50.):

    # disable learning
    if plasticity_snr:
        StrD1_SNr.enable_learning()
    else:
        StrD1_SNr.disable_learning()

    # calculate motor input
    motor_inputs = np.arange(0, 360, step=parameters['motor_step_size'])

    mean_d1_activities = []
    # simulate movement
    for motor_input in motor_inputs:

        input_m1 = circ_gauss(mu=motor_input, sigma=parameters['sig_m1'],
                              n=parameters['dim_motor'], scal=parameters['motor_step_size'], norm=False)

        # simulation state
        SNc.firing = 0
        PM.baseline = 0
        S1.baseline = 0
        CM.baseline = 0
        ann.simulate(t_wait)

        # set inputs
        SNc.firing = 1
        S1.baseline = 1.0
        CM.baseline = input_m1
        ann.simulate(sim_time)

        mean_d1_activities.append(np.mean(StrD1.r, axis=2))

    return np.array(mean_d1_activities)
