import os.path

import ANNarchy
import numpy as np

from kinematics.planar_arms import PlanarArms

from network.params import parameters
from network.model import *
from network.utils import bivariate_gauss, gauss

from monitoring import PopMonitor

from make_inputs import make_inputs

folder = 'test_1D_model/'

pops_monitor = [PM, S1, StrD1, GPe, SNr, Cortex, VL, M1, SNc, Output_Pop]
plt_types = ['Matrix', 'Plot', 'Matrix', 'Bar', 'Bar', 'Bar', 'Bar', 'Bar', 'Line', 'Polar']

monitors = PopMonitor(pops_monitor, auto_start=False, sampling_rate=1.0)

input_pm, _, input_m1, distance1 = make_inputs(start_point=parameters['starting_points'][1],
                                               end_point=parameters['reaching_points'][1])

mu = 1
input_s1 = gauss(np.arange(0, parameters['dim_s1']), mu=mu, sigma=0.2)

ann.compile('annarchy/' + folder)

# init
SNc.firing = 0
ann.simulate(50)

# initialize movement
monitors.start()

PM.baseline = input_pm
S1.baseline = input_s1
Cortex.baseline = input_m1

SNc.firing = 1

ann.simulate(int(distance1*2))

# goal reached
SNc.firing = 0
PM.baseline = 0
S1.baseline = 0
Cortex.baseline = 0
ann.simulate(100.)

# Second goal
input_pm, _, input_m1, distance2 = make_inputs(start_point=parameters['starting_points'][2],
                                               end_point=parameters['reaching_points'][2])

mu = 2
input_s1 = gauss(np.arange(0, parameters['dim_s1']), mu=mu, sigma=0.2)

PM.baseline = input_pm
S1.baseline = input_s1
Cortex.baseline = input_m1
SNc.firing = 1

ann.simulate(int(distance2*2))

# goal reached
SNc.firing = 0
PM.baseline = 0
S1.baseline = 0
Cortex.baseline = 0
ann.simulate(100.)

# save
folder = 'videos/'
if not os.path.exists(folder):
    os.makedirs('videos/')

monitors.animate_rates(plot_order=(5, 2), fig_size=(10, 5),
                       plot_types=plt_types)
