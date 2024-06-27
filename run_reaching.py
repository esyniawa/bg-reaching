import sys

from network.model import *
from monitoring import PopMonitor, ConMonitor
from make_inputs import train_position, test_movement

init_position = np.array((0, 150))
pops_monitor = [PM, S1, StrD1, GPe, SNr, CM, VL, M1, SNc, Output_Pop]

if __name__ == '__main__':

    sim_id, N_training_trials = sys.argv[1], int(sys.argv[2])

    # init monitors
    folder = f'run_model_{sim_id}/'
    training_monitors = PopMonitor(pops_monitor, auto_start=False, sampling_rate=50.)
    training_cons = ConMonitor([PM_StrD1, StrD1_SNr, M1_StrD1],
                               reshape_pre=[False, True, False],
                               reshape_post=[True, False, True])

    test_monitors = PopMonitor(pops_monitor, auto_start=False, sampling_rate=1.0)

    # compile model
    ann.compile('annarchy/' + folder)

    # training
    training_monitors.start()
    training_cons.extract_weights()

    positions = []
    for trial in range(N_training_trials):
        positions.append(init_position)
        init_position = train_position(init_position=init_position)

    # save
    # rates
    training_monitors.save(folder='results/' + 'training_' + folder, delete=True)
    # weights
    training_cons.extract_weights()
    training_cons.save_cons(folder='results/' + 'training_' + folder)
    training_cons.reset()
    # positions
    np.save('results/' + 'training_' + folder + 'learned_positions.npy', np.array(positions))

    # testing condition
    test_monitors.start()
    test_movement()

    # save data
    test_monitors.save(folder='results/' + 'test_' + folder, delete=True)

