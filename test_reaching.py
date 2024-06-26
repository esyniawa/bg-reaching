import numpy as np

from network.model import *
from monitoring import PopMonitor, ConMonitor
from make_inputs import sim_movement_m1_input


def test_closed_loop(plasticity: bool = True):

    # compile model
    folder = f'test_model/'
    ann.compile('annarchy/' + folder)

    # init monitors
    pops_monitor = [StrD1, M1, SNr, VL]
    monitors = PopMonitor(pops_monitor, auto_start=True, sampling_rate=2.)
    monitors_cons = ConMonitor([StrD1_SNr])

    # training
    monitors_cons.extract_weights()

    mean_d1 = sim_movement_m1_input(plasticity_snr=plasticity)

    monitors.animate_current_monitors(
        plot_types=['Matrix', 'Bar', 'Bar', 'Bar']
    )

    # save monitors
    monitors.save(folder='results/' + folder)

    monitors_cons.extract_weights()
    monitors_cons.save_cons(folder='results/' + folder)

    np.save('results/' + folder + 'mean_d1.npy', mean_d1)


if __name__ == '__main__':
    test_closed_loop(plasticity=False)
