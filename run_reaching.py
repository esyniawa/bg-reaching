import os.path

import ANNarchy
import numpy as np

from kinematics.planar_arms import PlanarArms

from network.params import parameters
from network.model import *
from network.utils import bivariate_gauss, gauss

from monitoring import PopMonitor

from make_inputs import train_position, test_movement

N_training_trials = 200
arms = PlanarArms(init_angles_right=np.array((20, 20)),
                  init_angles_left=np.array((20, 20)),
                  radians=False)

init_position = arms.end_effector_right[-1]
pops_monitor = [PM, S1, StrD1, GPe, SNr, Cortex, VL, M1, SNc, Output_Pop]

# init monitors
folder = 'test_model/'
if not os.path.exists('results/' + folder):
    os.makedirs('results/' + folder)


if __name__ == '__main__':

    monitors = PopMonitor(pops_monitor, auto_start=False, sampling_rate=1.0)

    # compile model
    ann.compile('annarchy/' + folder)

    # training
    for trial in range(N_training_trials):
        init_position = train_position(init_position=init_position)


    # testing condition
    monitors.start()
    test_movement()

    monitors.save(folder='results/' + folder, delete=True)
