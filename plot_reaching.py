import matplotlib.pyplot as plt
import numpy as np

from monitoring import PopMonitor

sim_id = 2
plot_weights = True
plot_positions = False
animate_str = True

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
