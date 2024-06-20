from network.model import *
from monitoring import PopMonitor, ConMonitor
from make_inputs import sim_movement_m1_input


def test_closed_loop(arm: str, plasticity: bool = True):

    # compile model
    folder = f'test_model/'
    ann.compile('annarchy/' + folder)

    # init monitors
    pops_monitor = [S1, StrD1, M1, GPe, SNr, VL, SNc, Output_Pop]
    monitors = PopMonitor(pops_monitor, auto_start=True, sampling_rate=2.)
    monitors_cons = ConMonitor([StrD1_SNr])

    # training
    monitors_cons.extract_weights()

    sim_movement_m1_input(arm, plasticity=plasticity)

    monitors.animate_rates(
        plot_types=['Matrix', 'Matrix', 'Bar', 'Bar', 'Bar', 'Bar', 'Line', 'Polar'],
        save_name='test.gif'
    )

    # save monitors
    monitors.save(folder='results/' + folder)

    monitors_cons.extract_weights()
    monitors_cons.save_cons(folder='results/' + folder)


if __name__ == '__main__':
    test_closed_loop(arm='right', plasticity=True)
