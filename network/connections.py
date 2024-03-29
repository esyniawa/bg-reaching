import numpy as np


def columnwise_connection(preDim: int,
                          postDim: tuple[int, int] | list[int, int],
                          weight=1.0):

    i, j = postDim
    w = np.array([[[None]*preDim]*j]*i)

    for pre_i in range(preDim):
        for post_i in range(i):
            w[post_i, pre_i, pre_i] = weight

    return w.reshape(i*j, preDim)
