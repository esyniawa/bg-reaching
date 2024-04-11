import ANNarchy as ann
import matplotlib.pyplot as plt
import numpy as np
import os


def ceil(a: float, precision: int = 0):
    return np.true_divide(np.ceil(a * 10**precision), 10**precision)


class PopMonitor(object):
    def __init__(self, populations: tuple | list,
                 variables: tuple | list | None = None,
                 sampling_rate: float = 2.0,
                 auto_start: bool = False):

        # define variables to track
        if variables is not None:
            assert len(populations) == len(variables), "The Arrays of populations and variables must have the same length"

            self.variables = variables
        else:
            self.variables = ['r'] * len(populations)

        # init monitors
        self.monitors = []

        for i, pop in enumerate(populations):
            self.monitors.append(ann.Monitor(pop, self.variables[i], period=sampling_rate, start=auto_start))

    def start(self):
        for monitor in self.monitors:
            monitor.start()

    def stop(self):
        for monitor in self.monitors:
            monitor.pause()

    def resume(self):
        for monitor in self.monitors:
            monitor.resume()

    def get(self, delete: bool = True, reshape: bool = True):
        res = {}

        for i, monitor in enumerate(self.monitors):
            res[self.variables[i] + '_' + monitor.object.name] = monitor.get(self.variables[i],
                                                                             keep=not delete, reshape=reshape)

        return res

    def save(self, folder, delete: bool = True):
        if not os.path.exists(folder):
            os.makedirs(folder)

        for i, monitor in enumerate(self.monitors):
            rec = monitor.get(self.variables[i], keep=not delete, reshape=True)
            np.save(folder + self.variables[i] + '_' + monitor.object.name, rec)

    def load(self, folder):
        monitor_dict = {}

        for i, monitor in enumerate(self.monitors):
            monitor_dict[self.variables[i] + '_' + monitor.object.name] = np.load(
                folder + self.variables[i] + '_' + monitor.object.name + '.npy')

        return monitor_dict

    @staticmethod
    def _reshape(m: np.ndarray, dim: int = 2):
        """
        Reshapes matrix m into a desired dim-dimensional array
        :param m:
        :param dim:
        :return:
        """
        shape = m.shape

        for i in range(m.ndim, dim, -1):
            new_shape = list(shape[:-1])
            new_shape[-1] = shape[-1] * shape[-2]
            shape = new_shape

        return m.reshape(shape)

    def plot_rates(self, plot_order: tuple[int, int],
                   plot_type: str = 'Plot',
                   fig_size: tuple[float, float] | list[float, float] = (5, 5),
                   save_name: str = None) -> None:

        """
        Plots 2D populations rates.
        :param plot_type: can be 'Plot' or 'Matrix'
        :param plot_order:
        :param fig_size:
        :param save_name:
        :return:
        """

        ncols, nrows = plot_order
        results = self.get(delete=False, reshape=True)

        fig = plt.figure(figsize=fig_size)
        for i, key in enumerate(results):
            plt.subplot(nrows, ncols, i + 1)
            if plot_type == 'Plot':
                if results[key].ndim > 2:
                    results[key] = PopMonitor._reshape(results[key])

                plt.plot(results[key])
                plt.ylabel('Activity')
                plt.xlabel(self.variables[i], loc='right')

            elif plot_type == 'Matrix':
                if results[key].ndim > 3:
                    results[key] = PopMonitor._reshape(results[key], dim=3)

                res_max = np.amax(abs(results[key]))
                img = plt.contourf(results[key].T, cmap='RdBu', vmin=-res_max, vmax=res_max)
                plt.colorbar(img, label=self.variables[i], orientation='horizontal')
                plt.xlabel('t', loc='right')

            plt.title(self.monitors[i].object.name, loc='left')

        if save_name is None:
            plt.show()
        else:
            folder, _ = os.path.split(save_name)
            if folder and not os.path.exists(folder):
                os.makedirs(folder)

            plt.savefig(save_name)
            plt.close(fig)

    def animate_rates(self,
                      plot_order: tuple[int, int],
                      plot_types: str | list | tuple = 'Bar',
                      fig_size: tuple[float, float] | list[float, float] = (5, 5),
                      t_init: int = 0,
                      save_name: str = None,
                      frames_per_sec: int | None = 10):

        from matplotlib.widgets import Slider
        import matplotlib.animation as animation

        ncols, nrows = plot_order
        results = self.get(delete=False, reshape=True)

        if isinstance(plot_types, str):
            plot_types = [plot_types] * len(results)

        fig = plt.figure(figsize=fig_size)
        ls = []

        for i, key in enumerate(results):

            # define type of the plot
            plot_type = plot_types[i]

            if plot_type == 'Matrix':

                ax = fig.add_subplot(nrows, ncols, i + 1)
                ax.set_title(self.monitors[i].object.name, loc='left')

                if results[key].ndim > 3:
                    results[key] = PopMonitor._reshape(results[key], dim=3)
                res_max = np.amax(abs(results[key]))

                l = ax.imshow(results[key][t_init], vmin=-res_max, vmax=res_max, cmap='RdBu',
                              origin='lower', interpolation='none')

            elif plot_type == 'Bar':

                ax = fig.add_subplot(nrows, ncols, i + 1)
                ax.set_title(self.monitors[i].object.name, loc='left')

                if results[key].ndim > 2:
                    results[key] = PopMonitor._reshape(results[key])

                res_max = np.amax(results[key])

                # plotting
                l = ax.bar(x=np.arange(1, results[key].shape[1] + 1, 1), height=results[key][t_init], width=0.5)

                ax.set_ylabel('Activity')
                ax.set_xlabel(self.variables[i], loc='right')
                ax.set_ylim([0, ceil(res_max + 0.1, precision=1)])

            elif plot_type == 'Plot':

                ax = fig.add_subplot(nrows, ncols, i + 1)
                ax.set_title(self.monitors[i].object.name, loc='left')

                if results[key].ndim > 3:
                    results[key] = PopMonitor._reshape(results[key], dim=3)

                res_max = np.amax(results[key])

                # plotting
                l = ax.plot(results[key][t_init])
                ax.set_ylabel('Activity')
                ax.set_xlabel(self.variables[i], loc='right')
                ax.set_ylim([0, ceil(res_max + 0.1, precision=1)])

            elif plot_type == 'Polar':

                ax = fig.add_subplot(nrows, ncols, i + 1, projection='polar')
                ax.set_title(self.monitors[i].object.name, loc='left')

                res_max = np.amax(np.sqrt(results[key][:, 1] ** 2 + results[key][:, 2] ** 2))

                if results[key].ndim > 2:
                    results[key] = PopMonitor._reshape(results[key])

                rad = (0, np.radians(results[key][t_init, 0]))
                r = (0, np.sqrt(results[key][t_init, 1] ** 2 + results[key][t_init, 2] ** 2))
                l = ax.plot(rad, r)
                ax.set_ylim([0, ceil(res_max + 0.1, precision=1)])

            elif plot_type == 'Line':

                ax = fig.add_subplot(nrows, ncols, i + 1)
                ax.set_title(self.monitors[i].object.name, loc='left')

                if results[key].ndim > 2:
                    results[key] = PopMonitor._reshape(results[key], dim=3)

                res_max = np.amax(results[key])

                # plotting
                ax.plot(results[key])
                l = ax.plot(results[key][t_init], marker='x', color='r')
                ax.set_ylabel('Activity')
                ax.set_xlabel('t', loc='right')
                ax.set_ylim([0, ceil(res_max + 0.1, precision=1)])

            elif plot_type is None:
                pass

            else:
                raise AssertionError('You must clarify which type of plot do you want!')

            ls.append((key, l, plot_type))

        # time length
        val_max = results[key].shape[0] - 1

        ax_slider = plt.axes((0.25, 0.05, 0.5, 0.03))
        time_slider = Slider(
            ax=ax_slider,
            label='n iteration',
            valmin=0,
            valmax=val_max,
            valinit=t_init
        )

        if save_name is None:
            def update(val):
                t = int(time_slider.val)
                time_slider.valtext.set_text(t)
                for key, plot, plt_type in ls:

                    if plt_type == 'Matrix':
                        plot.set_data(results[key][t, :, :])

                    elif plt_type == 'Plot':
                        if results[key].ndim == 3:
                            for j, line in plot:
                                line.set_ydata(results[key][t, :])
                        else:
                            plot[0].set_ydata(results[key][t, :])

                    elif plt_type == 'Bar':
                        for j, bar in enumerate(plot):
                            bar.set_height(results[key][t, j])

                    elif plt_type == 'Polar':
                        for line in plot:
                            line.set_xdata((0, np.radians(results[key][t, 0])))
                            line.set_ydata((0, np.sqrt(results[key][t, 1] ** 2 + results[key][t, 2] ** 2)))

                    elif plt_type == 'Line':
                        plot[0].set_ydata(results[key][t])
                        plot[0].set_xdata(t)

            time_slider.on_changed(update)

            plt.show()
        else:
            def update_animate(t):
                time_slider.valtext.set_text(t)
                time_slider.val = t
                subplots = []
                for key, plot, plt_type in ls:
                    subplots.append(plot)

                    if plt_type == 'Matrix':
                        plot.set_data(results[key][t])

                    elif plt_type == 'Plot':
                        if results[key].ndim == 3:
                            for j, line in plot:
                                line.set_ydata(results[key][t, :])
                        else:
                            plot[0].set_ydata(results[key][t, :])

                    elif plt_type == 'Bar':
                        for j, bar in enumerate(plot):
                            bar.set_height(results[key][t, j])

                    elif plt_type == 'Polar':
                        for line in plot:
                            line.set_xdata((0, np.radians(results[key][t, 0])))
                            line.set_ydata((0, np.sqrt(results[key][t, 1] ** 2 + results[key][t, 2] ** 2)))

                    elif plt_type == 'Line':
                        plot[0].set_ydata(results[key][t])
                        plot[0].set_xdata(t)

                return subplots

            folder, _ = os.path.split(save_name)
            if folder and not os.path.exists(folder):
                os.makedirs(folder)

            ani = animation.FuncAnimation(fig, update_animate, frames=np.arange(0, val_max))

            if save_name[-3:] == 'mp4':
                writer = animation.FFMpegWriter(fps=frames_per_sec)
            else:
                writer = animation.PillowWriter(fps=frames_per_sec)

            ani.save(save_name, writer=writer)
            plt.close(fig)

    def weight_difference(self,
                          plot_order: tuple[int, int],
                          fig_size: tuple[float, float] | list[float, float],
                          save_name: str = None):
        """
        Plots weight difference with imshow. Weights are normally recorded in [time, pre_synapse, post_synapse].
        :param plot_order:
        :param fig_size:
        :param save_name:
        :return:
        """

        ncols, nrows = plot_order
        results = self.get(delete=False)

        fig = plt.figure(figsize=fig_size)
        for i, key in enumerate(results):
            plt.subplot(ncols, nrows, i + 1)

            difference = results[key][-1, :, :] - results[key][0, :, :]
            x_axis = np.arange(0, len(difference.T))
            plt.plot(difference.T)
            plt.xticks(x_axis, x_axis+1)
            plt.xlabel('pre-synaptic neuron', loc='right')
            plt.title(key + ' : Weight difference', loc='left')

        if save_name is None:
            plt.show()
        else:
            folder, _ = os.path.split(save_name)
            if folder and not os.path.exists(folder):
                os.makedirs(folder)

            plt.savefig(save_name)
            plt.close(fig)


class ConMonitor(object):
    def __init__(self, connections):
        self.connections = connections
        self.weight_monitors = {}
        for con in connections:
            self.weight_monitors[con.name] = []

    def extract_weights(self):
        for con in self.connections:
            weights = np.array([dendrite.w for dendrite in con])
            self.weight_monitors[con.name].append(weights)

    def save_cons(self, folder: str):
        if not os.path.exists(folder):
            os.makedirs(folder)

        for con in self.connections:
            np.save(folder + 'w_' + con.name, self.weight_monitors[con.name])

    def load_cons(self, folder: str):
        con_dict = {}
        for con in self.connections:
            con_dict['w_' + con.name] = np.load(folder + 'w_' + con.name + '.npy')

        return con_dict

    def reset(self):
        for con in self.connections:
            self.weight_monitors[con.name] = []
