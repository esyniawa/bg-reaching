import sys

from kinematics.planar_arms import PlanarArms
from network.model import *
from monitoring import PopMonitor
from make_inputs import train_position, test_movement

N_training_trials = 1_000
arms = PlanarArms(init_angles_right=np.array((20, 20)),
                  init_angles_left=np.array((20, 20)),
                  radians=False)

init_position = arms.end_effector_right[-1]
pops_monitor = [PM, S1, StrD1, GPe, SNr, CM, VL, M1, SNc, Output_Pop]

if __name__ == '__main__':

    sim_id = sys.argv[1]

    # init monitors
    folder = f'run_model_{sim_id}/'
    training_monitors = PopMonitor(pops_monitor, auto_start=False, sampling_rate=200.)
    test_monitors = PopMonitor(pops_monitor, auto_start=False, sampling_rate=1.0)

    # compile model
    ann.compile('annarchy/' + folder)

    # training
    training_monitors.start()
    positions = []
    for trial in range(N_training_trials):
        positions.append(init_position)
        init_position = train_position(init_position=init_position)

    # save
    training_monitors.save(folder='results/' + 'training_' + folder, delete=True)

    # testing condition
    test_monitors.start()
    test_movement()

    # save data
    test_monitors.save(folder='results/' + 'test_' + folder, delete=True)
    np.save('results/' + 'test_' + folder + 'learned_positions.npy', np.array(positions))
