import matplotlib.pyplot as plt
import numpy as np

from monitoring import PopMonitor

sim_id = 1
plot_weights = False
plot_positions = False
animate_rates = False
plot_weights_diff = True

if plot_weights:
    weight_names = ['w_D1_SNr.npy', 'w_PM_D1.npy']

    for weight_name in weight_names:
        data_path = f'results/training_run_model_{sim_id}/' + weight_name
        data = np.load(data_path)

        fig, ax = plt.subplots()
        ax.imshow(data[1] - data[0], vmin=0, vmax=np.amax(data), cmap='Blues')

        ax.set_xlabel('pre-synaptic neuron', loc='right')
        ax.set_ylabel('post-synaptic neuron', loc='top')
        ax.set_aspect(0.1)
        ax.set_title('Weight difference: ' + weight_name)

        plt.show()
        plt.close(fig)

if plot_positions:
    from matplotlib.widgets import Slider
    from network.params import *
    from network.utils import bivariate_gauss

    animate = False

    data_path = f'results/test_run_model_{sim_id}/learned_positions.npy'
    data = np.load(data_path)

    learned_positions = data.shape[0]
    map = np.zeros(list(state_space.shape[:-1]) + [data.shape[0]])

    for i in range(learned_positions):
        map[:, :, i] = bivariate_gauss(xy=state_space,
                                       mu=data[i], sigma=parameters['sig_pm'], norm=True)

    # heatmap
    fig = plt.figure()
    heatmap = np.sum(map, axis=2)

    plt.imshow(heatmap, vmin=0, vmax=np.amax(heatmap), cmap='Blues')
    plt.show()
    plt.close(fig)

    if animate:
        # time length
        t_max = learned_positions - 1
        fig = plt.figure(figsize=(12, 8))

        l = plt.imshow(map[:, :, 0], vmin=0, vmax=1, cmap='Blues')

        ax_slider = plt.axes((0.25, 0.05, 0.5, 0.03))
        time_slider = Slider(
            ax=ax_slider,
            label='n iteration',
            valmin=0,
            valmax=t_max,
            valinit=0
        )

        def update(val):
            t = int(time_slider.val)
            time_slider.valtext.set_text(t)
            l.set_data(map[:, :, t])

        time_slider.on_changed(update)

        plt.show()
        plt.close(fig)

if animate_rates:
    from monitoring import PopMonitor

    training = True
    if training:
        folder = f'results/training_run_model_{sim_id}/'
    else:
        folder = f'results/test_run_model_{sim_id}/'

    populations = ['r_PM', 'r_S1', 'r_StrD1', 'r_SNr', 'r_VL', 'r_M1']
    plot_types = ['Matrix', 'Matrix', 'Matrix', 'Bar', 'Bar', 'Bar']

    PopMonitor.load_and_animate(folder=folder,
                                pops=populations,
                                plot_types=plot_types)

if plot_weights_diff:
    from monitoring import ConMonitor

    folder = f'results/training_run_model_{sim_id}/'

    connections = ['w_D1_SNr', 'w_proj12']
    ConMonitor.load_and_plot_wdiff(folder=folder, cons=connections, fig_size=(30, 20))

