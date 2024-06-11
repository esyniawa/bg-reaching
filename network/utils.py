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


def multivariate_gauss(xy: np.ndarray,
                       mu: np.ndarray,
                       sigma: np.ndarray,
                       norm: bool = False) -> np.ndarray:

    # check if dimensions are correct
    dim = mu.shape[0]
    if dim != sigma.shape[0] or dim != sigma.shape[1]:
        raise ValueError("Mu must be a vector with the dimensions n x 1 and "
                         "sigma must be a matrix with dimensions n x n.")

    det_sigma = np.linalg.det(sigma)

    if np.all(np.linalg.eigvals(sigma) > 0):
        inv_sigma = np.linalg.inv(sigma)
    else:
        raise ValueError("Sigma matrix must be positive definite.")

    exp = np.einsum('...k,kl,...l->...', xy - mu, inv_sigma, xy - mu)

    if norm:
        return np.exp(-0.5 * exp) / np.sqrt((2*np.pi) ** dim * det_sigma)
    else:
        return np.exp(-0.5 * exp)


def bivariate_gauss(mu: tuple[float, float],
                    sigma: float,
                    xy: np.ndarray,
                    norm: bool = False, plot: bool = False, limit: float | None = None):

    a = multivariate_gauss(xy, mu=np.array(mu), sigma=sigma * np.eye(len(mu)), norm=norm)

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
