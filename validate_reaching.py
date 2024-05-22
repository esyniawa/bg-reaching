import sys

from network.model import *
from monitoring import PopMonitor, ConMonitor
from make_inputs import train_position, test_movement, train_fixed_position

init_position = np.array((0, 150))
pops_monitor = [PM, S1, StrD1, GPe, SNr, CM, VL, M1, SNc, Output_Pop]

if __name__ == '__main__':

    sim_id, N_training_trials = sys.argv[1], int(sys.argv[2])

    init_positions = [
        np.array((-100, 200)),
        np.array((-100, 50)),
        np.array((100, 50)),
        np.array((100, 200)),
    ]

    goals = [
        np.array((-100, 50)),
        np.array((100, 50)),
        np.array((100, 200)),
        np.array((-100, 200))
    ]

    # init monitors
    folder = f'validate_model_{sim_id}/'
    training_monitors = PopMonitor(pops_monitor, auto_start=False, sampling_rate=200.)
    training_cons = ConMonitor([PM_StrD1, StrD1_SNr])

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

    for start, end in zip(init_positions, goals):
        train_fixed_position(init_position=start, goal=end)
        
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

