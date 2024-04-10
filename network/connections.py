import numpy as np


def column_wise_connection(preDim: int,
                           postDim: tuple[int, int] | list[int, int],
                           weight=1.0):

    i, j = postDim
    w = np.array([[[None]*preDim]*j]*i)

    for pre_i in range(preDim):
        w[:, pre_i, pre_i] = weight

    return w.reshape(i*j, preDim)


def row_wise_connection(preDim: int,
                        postDim: tuple[int, int] | list[int, int],
                        weight=1.0):

    i, j = postDim
    w = np.array([[[None]*preDim]*j]*i)

    for pre_i in range(preDim):
        w[pre_i, :, pre_i] = weight

    return w.reshape(i*j, preDim)


def w_ones_to_all(preDim: list | tuple, postDim: list | tuple, allDim: int = -1, weight: float = 1.0):
    """
    Returns connection matrix, where all layers are connected one-to-one, except one layer (allDim), which is connected
    all-to-all.
    :param preDim: Shape of pre-synaptic layer
    :param postDim: Shape of post-synaptic layer
    :param allDim:  Index of the layer on which the all-to-all connection takes place.
    :param weight: Connection strength
    :return:
    """
    if isinstance(preDim, tuple):
        preDim = list(preDim)
    if isinstance(postDim, tuple):
        postDim = list(postDim)

    pre_k, post_k = preDim.pop(allDim), postDim.pop(allDim)

    if not preDim == postDim:
        raise AssertionError

    n = np.prod(preDim)
    w = np.array([[[[None] * post_k] * n] * pre_k] * n)

    for i in range(n):
        for k_pre in range(pre_k):
            for k_post in range(post_k):
                w[i, k_pre, i, k_post] = weight

    return w.reshape((n * pre_k, n * post_k)).T


def w_pooling(preDim: tuple | list, poolingDim: int = -1, weight: float = 1.0):
    """
    Just a summation pooling connection. Implemented because ANNarchys Pooling Function doesn't support summation.
    :param preDim: Shape of pre-synaptic layer
    :param poolingDim: Index of the layer on which summation takes place.
    :param weight: Connection strength
    :return:
    """

    if isinstance(preDim, tuple):
        preDim = list(preDim)

    pre_k = preDim.pop(poolingDim)

    n = np.prod(preDim)
    w = np.array([[[None] * pre_k] * pre_k] * n)

    for i in range(n):
        for k_pre in range(pre_k):
            w[i, k_pre, k_pre] = weight

    return w.reshape((n * pre_k, pre_k)).T

def pop_code_output(preferred_angles: np.ndarray, radians: bool = False):
    """
    This matrix reads out a population code in which neurons prefer a certain angle in space. The first neuron encodes
    the resulting movement angles from the population activity, the second the x component and the third the y component
    of the movement. The movement angle must also be normalized (not part of this connection matrix).
    :param preferred_angles: Preferred angle of the neurons
    :param radians: Specification of the angles in radians?
    :return w: Resulting connectivity matrix
    """

    pre_shape = preferred_angles.shape[0]
    w = np.array([[None]*pre_shape]*3)

    # angles
    for i, angle in enumerate(preferred_angles):
        if radians:
            w[0, i] = float(np.degrees(angle))
        else:
            w[0, i] = float(angle)

    # x component
    for i, angle in enumerate(preferred_angles):
        if radians:
            w[1, i] = np.cos(angle)
        else:
            w[1, i] = np.cos(np.radians(angle))

    # y component
    for i, angle in enumerate(preferred_angles):
        if radians:
            w[2, i] = np.sin(angle)
        else:
            w[2, i] = np.sin(np.radians(angle))

    return w
