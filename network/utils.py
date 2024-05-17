import numpy as np


def circ_gauss(mu: float,
               sigma: float,
               n: int,
               scal: float = 1.0,
               norm: bool = False,
               limit: float | None = None,
               plot: bool = False):

    mu /= scal
    ret = np.zeros(n)
    for i in range(n):
        res = abs(i-mu)
        if res > n/2:
            ret[i] = n - res
        else:
            ret[i] = res

    ret = np.exp(-np.power(scal * ret / sigma, 2) / 2)

    if norm:
        ret /= (np.sqrt(2.0 * np.pi) * sigma)

    if limit is not None:
        ret[ret < limit] = 0.0

    if plot:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(ret)
        plt.xticks([])
        plt.yticks([])
        plt.show()

    return ret


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


if __name__ == '__main__':
    circ_gauss(0, 25, 22, scal=15, plot=True)
