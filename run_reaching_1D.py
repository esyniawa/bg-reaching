import numpy as np

from kinematics.planar_arms import PlanarArms

from network.params import parameter_1D
from network.model_1D import *
from network.utils import bivariate_gauss, gauss

from monitoring import PopMonitor

from make_inputs import make_inputs

folder = 'test_1D_model/'

rates = PopMonitor([PM, S1, StrD1, SNr, STN, VL, M1, SNc, Out], auto_start=True, sampling_rate=10)
# weights = PopMonitor([PM_StrD1, StrD1_SNr, STN_SNr], variables=['w', 'w', 'w'], sampling_rate=50., auto_start=True)


input_pm, _, input_stn, distance = make_inputs(start_point=parameter_1D['starting_points'][1],
                                               end_point=parameter_1D['reaching_points'][1])

input_s1 = gauss(np.arange(0, parameter_1D['dim_s1']), mu=0, sigma=0.5)

ann.compile('annarchy/' + folder)


PM.baseline = input_pm
S1.baseline = input_s1
STN.baseline = input_stn
SNc.firing = 1

ann.simulate(2000.)

SNc.firing = 0

ann.simulate(500.)

rates.save(folder='rates/', delete=False)
# weights.save(folder='rates/', delete=False)

rates.animate_rates(plot_order=(3, 3),
                    plot_types=['Matrix', 'Plot', 'Matrix', 'Matrix', 'Bar', 'Matrix', 'Matrix', 'Bar', 'Polar'],
                    fig_size=(15, 10),
                    save_name='test3')
