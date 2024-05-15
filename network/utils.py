import numpy as np


def gauss(x: np.ndarray, mu: float, sigma: float, norm: bool = True, limit: float | None = None, plot: bool = False):
    """

    :param x:
    :param mu: Mean
    :param sigma: Standard deviation
    :param norm: Should the function be normalized to 1.0?
    :param limit: Clipping function: If a value undercuts the "limit", it is set to 0.0
    :param plot: Should the function be plotted?
    :return: Normal(x)
    """

    res = np.exp(-np.power((x - mu) / sigma, 2) / 2)

    if norm:
        res /= (np.sqrt(2.0 * np.pi) * sigma)

    if limit is not None:
        res[res < limit] = 0.0

    if plot:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(res)
        plt.xticks([])
        plt.yticks([])
        plt.show()

    return res


def distance(n: int, index: int, scal: float = 1.0):
    ret = np.zeros(n)
    for i in range(n):
        res = abs(i-index)
        if res > n/2:
            ret[i] = n - res
        else:
            ret[i] = res
    return scal * ret


def create_state_space(x_bound: tuple[int, int],
                       y_bound: tuple[int, int],
                       step_size_x: float,
                       step_size_y: float,
                       resolution_factor=1.0):

    x_lowerbound, x_upperbound = x_bound
    y_lowerbound, y_upperbound = y_bound

    x = np.arange(start=x_lowerbound, stop=x_upperbound+step_size_x, step=np.ceil(resolution_factor * step_size_x))
    y = np.arange(start=y_lowerbound, stop=y_upperbound+step_size_y, step=np.ceil(resolution_factor * step_size_y))

    xx, yy = np.meshgrid(x, y)
    xy = np.dstack((xx, yy))

    return xy


def bivariate_gauss(mu: tuple[float, float],
                    sigma: float,
                    xy: np.ndarray,
                    norm: bool = False, plot: bool = False, limit: float | None = None):

    from scipy.stats import multivariate_normal

    rv = multivariate_normal(mu, cov=sigma * np.identity(2))
    a = rv.pdf(xy)

    if norm:
        a /= np.max(a)

    if limit is not None:
        a[a < limit] = np.NaN

    if plot:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)
        img = ax.contourf(a, cmap='Purples')
        plt.xticks([])
        plt.yticks([])
        plt.colorbar(img)
        plt.show()

    return a
