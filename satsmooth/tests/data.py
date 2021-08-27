import numpy as np
from skimage.exposure import rescale_intensity


def create_1d(seed1, seed2, sigma, shift, n):

    x1 = np.float64(np.linspace(0, n * 4, n))

    np.random.seed(seed1)
    y = w = np.random.normal(size=n)

    for t in range(n):
        y[t] = y[t - 1] + w[t]

    np.random.seed(seed2)
    y += sigma * np.random.normal(size=n)

    # Scale the data
    y = rescale_intensity(y, out_range=(0.1, 0.8))
    y += shift

    x2 = np.float64(np.arange(x1[0], x1[-1]))
    y = np.float32(y)

    return x1, x2, y
