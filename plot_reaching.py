import matplotlib.pyplot as plt
import numpy as np

sim_id = 1
plot_weights = True

if plot_weights:
    weight_names = ['w_D1_SNr.npy', 'w_PM_D1.npy']

    for weight_name in weight_names:
        data_path = f'results/training_run_model_{sim_id}/' + weight_name
        data = np.load(data_path)


        fig, ax = plt.subplots()

        ax.imshow(data[1] - data[0], vmin=0, vmax=np.amax(data), cmap='Blues')

        ax.set_xlabel('pre-synaptic neuron', loc='right')
        ax.set_ylabel('post-synaptic neuron', loc='top')

        ax.set_title('Weight difference: ' + weight_name)

        plt.show()
        plt.close(fig)
