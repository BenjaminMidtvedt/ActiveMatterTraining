import numpy as np


def autocorrelation(x):
    """A fast implementation of autocorrelation using fourier transforms"""
    N = len(x)
    f_transform = np.fft.fft(x, n=2 * N)

    # I think this can be faster by using symmetries
    Fmag2 = f_transform * f_transform.conjugate()

    xp = np.fft.ifft(Fmag2)
    xp = xp[:N].real

    scale = np.arange(N + 1, 1, step=-1)

    return xp / scale


def MSD(x):
    """A fast implementation of MSD calculation using fourier transforms"""

    N = len(x)

    dist = np.square(x).sum(axis=1)
    dist = np.append(dist, 0)
    S2 = sum([autocorrelation(x[:, i]) for i in range(x.shape[1])])
    Q = 2 * dist.sum()

    S1 = np.zeros(N)
    for m in range(N):
        Q = Q - dist[m - 1] - dist[N - m]
        S1[m] = Q / (N - m)

    return S1 - 2 * S2
