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
    sim_times = []

    for start, end in zip(init_positions, goals):
        train_fixed_position(init_position=start, goal=end)

    for trial in range(N_training_trials):
        positions.append(init_position)
        init_position, sim_time = train_position(init_position=init_position, return_sim_time=True)
        sim_times.append(sim_time)

    # save
    # rates
    training_monitors.save(folder='results/' + 'training_' + folder, delete=True)
    # weights
    training_cons.extract_weights()
    training_cons.save_cons(folder='results/' + 'training_' + folder)
    training_cons.reset()
    # positions and sim times
    np.save('results/' + 'training_' + folder + 'learned_positions.npy', np.array(positions))
    np.save('results/' + 'training_' + folder + 'sim_times.npy', np.array(sim_times))

    # testing condition
    test_monitors.start()
    for scale in [1.0, 1.5, 2.0, 3.0, 5.0]:
        test_movement(scale_s1=scale)

    # save data
    test_monitors.save(folder='results/' + 'test_' + folder, delete=True)
