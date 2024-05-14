import os.path
import sys

import ANNarchy
import numpy as np

from kinematics.planar_arms import PlanarArms

from network.params import parameters
from network.model import *
from network.utils import bivariate_gauss, gauss

from monitoring import PopMonitor

from make_inputs import train_position, test_movement

N_training_trials = 1_000
arms = PlanarArms(init_angles_right=np.array((20, 20)),
                  init_angles_left=np.array((20, 20)),
                  radians=False)

init_position = arms.end_effector_right[-1]
pops_monitor = [PM, S1, StrD1, GPe, SNr, Cortex, VL, M1, SNc, Output_Pop]

if __name__ == '__main__':

    sim_id = sys.argv[1]

    # init monitors
    folder = f'run_model_{sim_id}/'
    if not os.path.exists('results/' + folder):
        os.makedirs('results/' + folder)

    monitors = PopMonitor(pops_monitor, auto_start=False, sampling_rate=1.0)

    # compile model
    ann.compile('annarchy/' + folder)

    # training
    positions = []
    for trial in range(N_training_trials):
        positions.append(init_position)
        init_position = train_position(init_position=init_position)

    # testing condition
    monitors.start()
    test_movement()

    # save data
    monitors.save(folder='results/' + folder, delete=True)
    np.save('results/' + folder + 'learned_positions.npy', np.array(positions))
