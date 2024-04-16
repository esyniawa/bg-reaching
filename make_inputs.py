import numpy as np

from network.params import parameter_1D, state_space
from network.utils import bivariate_gauss, gauss

from kinematics.planar_arms import PlanarArms


def make_inputs(start_point: list[float, float] | tuple[float, float],
                end_point: list[float, float] | tuple[float, float],
                show_input: bool = False):

    input_pm = bivariate_gauss(xy=state_space,
                               mu=end_point,
                               sigma=parameter_1D['sig_pm'],
                               norm=True,
                               plot=show_input)

    input_s1 = bivariate_gauss(xy=state_space,
                               mu=start_point,
                               sigma=parameter_1D['sig_s1'],
                               norm=True,
                               plot=show_input)

    motor_angle, distance = PlanarArms.calc_motor_vector(init_pos=np.array(start_point),
                                                         end_pos=np.array(end_point),
                                                         arm=parameter_1D['moving_arm'])

    input_m1 = 10 * gauss(parameter_1D['motor_orientations'], mu=motor_angle, sigma=parameter_1D['sig_stn'], norm=False)
    input_stn = np.amax(input_m1) - input_m1

    return input_pm, input_s1, input_m1, input_stn, distance


if __name__ == '__main__':
    make_inputs(start_point=parameter_1D['starting_points'][1], end_point=parameter_1D['reaching_points'][1],
                show_input=True)
