import ANNarchy as ann
import matplotlib.pyplot as plt
import numpy as np
import os


def ceil(a: float, precision: int = 0):
    """
    Calculate the ceiling value of a number 'a' with a specified precision.

    Parameters:
    :param a: The number for which the ceiling value needs to be calculated.
    :param precision: The number of decimal places to consider for precision (default is 0).

    :return: The ceiling value of 'a' with the specified precision.

    Example:
    ceil(3.14159, 2) returns 3.15
    ceil(5.67, 0) returns 6.0
    """
    return np.true_divide(np.ceil(a * 10**precision), 10**precision)


def find_largest_factors(c: int):
    """
    Returns the two largest factors a and b of an integer c, such that a * b = c.
    """
    for a in range(int(c**0.5), 0, -1):
        if c % a == 0:
            b = c // a
            return b, a
    return 1, c


class PopMonitor(object):
    def __init__(self, populations: tuple | list,
                 variables: tuple[str] | list[str] | None = None,
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

    def get_specific_monitor(self, pop_name: str, delete: bool = True, reshape: bool = True):
        index = [i for i, monitor in enumerate(self.monitors) if pop_name in monitor.object.name][0]

        ret = self.monitors[index].get(self.variables[index],
                                       keep=not delete,
                                       reshape=reshape)

        return ret

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

    def animate_rates(self,
                      plot_order: tuple[int, int] | None = None,
                      plot_types: str | list | tuple = 'Bar',
                      fig_size: tuple[float, float] | list[float, float] = (10, 10),
                      t_init: int = 0,
                      save_name: str = None,
                      label_ticks: bool = True,
                      frames_per_sec: int | None = 10):

        # TODO: Making a plot type class to trim the code

        from matplotlib.widgets import Slider
        import matplotlib.animation as animation

        # get results
        results = self.get(delete=False, reshape=True)

        # define plot layout
        if plot_order is None:
            ncols, nrows = find_largest_factors(len(results))
        else:
            ncols, nrows = plot_order

        # define plot types if not defined
        if isinstance(plot_types, str):
            plot_types = [plot_types] * len(results)

        # fill the figure
        fig = plt.figure(figsize=fig_size)
        subfigs = fig.subfigures(nrows, ncols)

        plot_lists = []
        for outer_i, (subfig, key) in enumerate(zip(subfigs.flat, results)):

            # assignment plot type + key
            plot_type = plot_types[outer_i]

            # set title
            subfig.suptitle(key)
            subfig.subplots_adjust(left=0.1, bottom=0.2, right=0.9, top=0.9, wspace=0.1, hspace=0.1)

            # good ol' switcharoo
            if plot_type == 'Matrix':
                if results[key].ndim > 4:
                    results[key] = PopMonitor._reshape(results[key])
                res_max = np.amax(abs(results[key]))

                # subplots
                if results[key].ndim == 4:
                    last_dim = results[key].shape[-1]
                    inner_rows, inner_cols = find_largest_factors(last_dim)

                    # add subsubplots
                    axs = subfig.subplots(inner_rows, inner_cols)
                    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.05, hspace=0.05)

                    plots = []
                    for ax, result in zip(axs.flat, np.rollaxis(results[key][t_init], -1)):
                        p = ax.imshow(result, vmin=-res_max, vmax=res_max, cmap='RdBu',
                                      origin='lower', interpolation='none')
                        # set off tick labels for better arrangement
                        ax.set_xticks([])
                        ax.set_yticks([])

                        plots.append(p)
                else:
                    ax = subfig.subplots()
                    plots = ax.imshow(results[key][t_init], vmin=-res_max, vmax=res_max, cmap='RdBu',
                                      origin='lower', interpolation='none')

            elif plot_type == 'Bar':

                if results[key].ndim > 3:
                    results[key] = PopMonitor._reshape(results[key])
                res_max = np.amax(abs(results[key]))

                # subplots
                if results[key].ndim == 3:
                    last_dim = results[key].shape[-1]
                    inner_rows, inner_cols = find_largest_factors(last_dim)

                    axs = subfig.subplots(inner_rows, inner_cols)
                    plots = []
                    for ax, result in zip(axs.flat, np.rollaxis(results[key][t_init], -1)):
                        p = ax.bar(x=np.arange(1, result.shape[1] + 1, 1), height=result, width=0.5)

                        ax.set_ylabel('Activity')
                        ax.set_xlabel(self.variables[outer_i], loc='right')
                        ax.set_ylim([0, ceil(res_max + 0.1, precision=1)])

                        plots.append(p)
                else:
                    ax = subfig.subplots()
                    plots = ax.bar(x=np.arange(1, results[key].shape[1] + 1, 1), height=results[key][t_init], width=0.5)

                    ax.set_ylabel('Activity')
                    ax.set_xlabel(self.variables[outer_i], loc='right')
                    ax.set_ylim([0, ceil(res_max + 0.1, precision=1)])

            elif plot_type == 'Polar':

                res_max = np.amax(np.sqrt(results[key][:, 1] ** 2 + results[key][:, 2] ** 2))
                ax = subfig.add_subplot(projection='polar')

                if results[key].ndim > 2:
                    results[key] = PopMonitor._reshape(results[key])

                rad = (0, np.radians(results[key][t_init, 0]))
                r = (0, np.sqrt(results[key][t_init, 1] ** 2 + results[key][t_init, 2] ** 2))
                plots = ax.plot(rad, r)
                ax.set_ylim([0, ceil(res_max + 0.1, precision=1)])
                ax.set_ylabel([])

            elif plot_type == 'Line':

                ax = subfig.subplots()

                if results[key].ndim > 2:
                    results[key] = PopMonitor._reshape(results[key], dim=3)

                res_max = np.amax(results[key])

                # plotting
                ax.plot(results[key])
                plots = ax.plot(results[key][t_init], marker='x', color='r')
                ax.set_ylabel('Activity')
                ax.set_xlabel('t', loc='right')
                ax.set_ylim([0, ceil(res_max + 0.1, precision=1)])

            elif plot_type is None:
                plots = None

            else:
                raise AssertionError('You must clarify which type of plot do you want!')

            if not label_ticks:
                plt.xticks([])
                plt.yticks([])

            plot_lists.append((key, plots, plot_type))

        # time length
        val_max = results[key].shape[0] - 1

        if save_name is None:

            ax_slider = plt.axes((0.2, 0.05, 0.5, 0.03))
            time_slider = Slider(
                ax=ax_slider,
                label='n iteration',
                valmin=0,
                valmax=val_max,
                valinit=t_init
            )

            def update(val):
                t = int(time_slider.val)
                time_slider.valtext.set_text(t)

                for key, subfigure, plt_type in plot_lists:

                    if plt_type == 'Matrix':
                        if isinstance(subfigure, list):
                            for plot, result in zip(subfigure, np.rollaxis(results[key], axis=-1)):
                                plot.set_data(result[t])
                        else:
                            subfigure.set_data(results[key][t])

                    elif plt_type == 'Bar':
                        if isinstance(subfigure, list):
                            for plot, result in zip(subfigure, np.rollaxis(results[key], axis=-1)):
                                for j, bar in enumerate(plot):
                                    bar.set_height(result[t, j])
                        else:
                            for j, bar in enumerate(subfigure):
                                bar.set_height(results[key][t, j])

                    elif plt_type == 'Polar':
                        for line in subfigure:
                            line.set_xdata((0, np.radians(results[key][t, 0])))
                            line.set_ydata((0, np.sqrt(results[key][t, 1] ** 2 + results[key][t, 2] ** 2)))

                    elif plt_type == 'Line':
                        subfigure[0].set_ydata(results[key][t])
                        subfigure[0].set_xdata(t)

            time_slider.on_changed(update)

            plt.show()
        else:
            def update_animate(t):
                subplots = []
                for key, subfigure, plt_type in plot_lists:

                    if plt_type == 'Matrix':
                        if isinstance(subfigure, list):
                            for plot, result in zip(subfigure, np.rollaxis(results[key], axis=-1)):
                                plot.set_data(result[t])
                        else:
                            subfigure.set_data(results[key][t])

                    elif plt_type == 'Bar':
                        if isinstance(subfigure, list):
                            for plot, result in zip(subfigure, np.rollaxis(results[key], axis=-1)):
                                for j, bar in enumerate(plot):
                                    bar.set_height(result[t, j])
                        else:
                            for j, bar in enumerate(subfigure):
                                bar.set_height(results[key][t, j])

                    elif plt_type == 'Polar':
                        for line in subfigure:
                            line.set_xdata((0, np.radians(results[key][t, 0])))
                            line.set_ydata((0, np.sqrt(results[key][t, 1] ** 2 + results[key][t, 2] ** 2)))

                    elif plt_type == 'Line':
                        subfigure[0].set_ydata(results[key][t])
                        subfigure[0].set_xdata(t)

            # make folder if not exists
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

    def animate_population_3D(self,
                              pop_name: str,
                              iter_dim: int,
                              plot_order: tuple,
                              t_init: int = 0,
                              fig_size: tuple[float, float] = (12, 8),
                              save_name: str = None,
                              label_ticks: bool = True,
                              frames_per_sec: int | None = 10):

        from matplotlib.widgets import Slider
        import matplotlib.animation as animation

        try:
            monitor = self.get_specific_monitor(pop_name=pop_name, delete=False)
        except:
            raise AssertionError(f'Population {pop_name} is not in the monitor list!')

        # time length
        t_max = monitor.shape[0] - 1
        results = monitor[0]
        val_max = np.amax(monitor)

        assert results.shape[iter_dim] <= np.prod(plot_order), ('There are not enough subplots to plot all dimensions '
                                                                'of the population!!!')

        ncols, nrows = plot_order
        fig = plt.figure(figsize=fig_size)
        ls = []

        for i, result in enumerate(np.rollaxis(results, iter_dim)):
            ax = fig.add_subplot(nrows, ncols, i + 1)
            l = ax.imshow(result, vmin=0, vmax=val_max, cmap='Blues')
            ls.append(l)

        if not label_ticks:
            plt.xticks([])
            plt.yticks([])

        if save_name is None:

            ax_slider = plt.axes((0.25, 0.05, 0.5, 0.03))
            time_slider = Slider(
                ax=ax_slider,
                label='n iteration',
                valmin=0,
                valmax=t_max,
                valinit=t_init
            )

            def update(val):
                t = int(time_slider.val)
                time_slider.valtext.set_text(t)
                results = monitor[t]

                for result, plot in zip(np.rollaxis(results, iter_dim), ls):
                    plot.set_data(result)

            time_slider.on_changed(update)
            plt.show()

        else:
            def update_animate(t):
                for result, plot in zip(np.rollaxis(monitor[int(t)], iter_dim), ls):
                    plot.set_data(result)
                    return ls

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
