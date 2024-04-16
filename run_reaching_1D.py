import os.path

import ANNarchy
import numpy as np

from kinematics.planar_arms import PlanarArms

from network.params import parameter_1D
from network.model_1D import *
from network.utils import bivariate_gauss, gauss

from monitoring import PopMonitor

from make_inputs import make_inputs

folder = 'test_1D_model/'

pops_monitor = [PM, S1, StrD1, SNr, STN, VL, M1, SNc, Out]
names = [pop.name for pop in pops_monitor]

plt_types = ['Matrix', 'Plot', 'Matrix', 'Matrix', 'Bar', 'Matrix', 'Matrix', 'Line', 'Polar']

monitors = [PopMonitor([pop], auto_start=False, sampling_rate=1.0) for pop in pops_monitor]

rates = PopMonitor(pops_monitor, auto_start=False, sampling_rate=1.0)

input_pm, _, input_m1, input_stn, distance1 = make_inputs(start_point=parameter_1D['starting_points'][1],
                                                         end_point=parameter_1D['reaching_points'][1])

mu = 1
input_s1 = gauss(np.arange(0, parameter_1D['dim_s1']), mu=mu, sigma=0.2)

ann.compile('annarchy/' + folder)

# init
StrD1_SNr.transmission = False
SNc.firing = 0
ann.simulate(50)

# initialize movement
StrD1_SNr.transmission = True

for monitor in monitors:
    monitor.start()
rates.start()

PM.baseline = input_pm
S1.baseline = input_s1
STN.baseline = input_stn
M1[mu, :].baseline = input_m1
SNc.firing = 1

ann.simulate(int(distance1*2))

# goal reached
SNc.firing = 0
PM.baseline = 0
S1.baseline = 0
STN.baseline = 0
M1[mu, :].baseline = 0
ann.simulate(100.)

input_pm, _, input_m1, input_stn, distance2 = make_inputs(start_point=parameter_1D['starting_points'][2],
                                                         end_point=parameter_1D['reaching_points'][2])

mu = 2
input_s1 = gauss(np.arange(0, parameter_1D['dim_s1']), mu=mu, sigma=0.2)

PM.baseline = input_pm
S1.baseline = input_s1
STN.baseline = input_stn
M1[mu, :].baseline = input_m1
SNc.firing = 1

ann.simulate(int(distance2*2))

# goal reached
SNc.firing = 0
PM.baseline = 0
S1.baseline = 0
STN.baseline = 0
M1[mu, :].baseline = 0

ann.simulate(100.)

folder = 'videos/'
if not os.path.exists(folder):
    os.makedirs('videos/')

#
for monitor, plot_type, name in zip(monitors, plt_types, names):
    monitor.animate_rates(plot_order=(1, 1), plot_types=plot_type, save_name=folder + name + '.gif', frames_per_sec=20)

# rates.animate_rates(plot_order=(3, 3), fig_size=(20, 18), plot_types=plt_types)

# animate movement
init_thetas = PlanarArms.inverse_kinematics(arm=parameter_1D['moving_arm'],
                                            end_effector=np.array(parameter_1D['starting_points'][1]),
                                            starting_angles=np.array((20,20)),
                                            radians=False)

myarms = PlanarArms(init_angles_left=np.array((0, 0)),
                    init_angles_right=init_thetas,
                    radians=True)

myarms.move_to_position(arm=parameter_1D['moving_arm'], end_effector=parameter_1D['reaching_points'][1], num_iterations=int(distance1*2))
myarms.wait(100)
myarms.move_to_position(arm=parameter_1D['moving_arm'], end_effector=parameter_1D['reaching_points'][2], num_iterations=int(distance2*2))
myarms.wait(100)

myarms.plot_trajectory(save_name = 'videos/trajectory.gif', frames_per_sec = 20,
                       points=(parameter_1D['starting_points'][1], parameter_1D['reaching_points'][1], parameter_1D['reaching_points'][2]))