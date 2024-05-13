import numpy as np

from network.params import parameters, state_space
from network.utils import bivariate_gauss, gauss
from network.model import *

from kinematics.planar_arms import PlanarArms


def make_inputs(start_point: list[float, float] | tuple[float, float],
                end_point: list[float, float] | tuple[float, float],
                show_input: bool = False):

    input_pm = bivariate_gauss(xy=state_space,
                               mu=end_point,
                               sigma=parameters['sig_pm'],
                               norm=True,
                               plot=show_input)

    input_s1 = bivariate_gauss(xy=state_space,
                               mu=start_point,
                               sigma=parameters['sig_s1'],
                               norm=True,
                               plot=show_input)

    motor_angle, distance = PlanarArms.calc_motor_vector(init_pos=np.array(start_point),
                                                         end_pos=np.array(end_point),
                                                         arm=parameters['moving_arm'])

    input_m1 = gauss(parameters['motor_orientations'], mu=motor_angle, sigma=parameters['sig_m1'], norm=False)

    return input_pm, input_s1, input_m1, distance


def train_position(init_position: np.ndarray,
                   scale_movement: float = 2.0,
                   t_wait: float = 20.) -> np.ndarray:

    random_x = np.random.uniform(low=parameters['x_reaching_space_limits'][0],
                                 high=parameters['x_reaching_space_limits'][1])
    random_y = np.random.uniform(low=parameters['y_reaching_space_limits'][0],
                                 high=parameters['y_reaching_space_limits'][1])

    base_pm, base_s1, base_m1, distance = make_inputs(start_point=init_position,
                                                      end_point=[random_x, random_y])

    # simulation state
    SNc.firing = 0
    PM.baseline = 0
    S1.baseline = 0
    Cortex.baseline = 0
    ann.simulate(t_wait)

    # set inputs
    SNc.firing = 1
    PM.baseline = base_pm
    S1.baseline = base_s1
    Cortex.baseline = base_m1
    ann.simulate(distance * scale_movement)

    return np.array([random_x, random_y])


def test_movement(scale_movement: float = 2.0, t_wait: float = 20.) -> None:

    points_to_follow = [
        np.array((-100, 200)),
        np.array((-100, 50)),
        np.array((100, 50)),
        np.array((100, 200)),
    ]

    n_points = len(points_to_follow)

    # simulate movement
    ann.disable_learning()

    for i, point in enumerate(points_to_follow):

        # make inputs for PM
        input_pm, input_s1, _, distance = make_inputs(start_point=point,
                                                      end_point=points_to_follow[(i+1) % n_points])

        # simulation state
        SNc.firing = 0
        PM.baseline = 0
        S1.baseline = 0
        Cortex.baseline = 0
        ann.simulate(t_wait)

        # set inputs
        SNc.firing = 1
        PM.baseline = input_pm
        S1.baseline = input_s1
        ann.simulate(distance * scale_movement)


if __name__ == '__main__':
    make_inputs(start_point=parameters['starting_points'][1], end_point=parameters['reaching_points'][1],
                show_input=True)
