import numpy as np
from scipy.interpolate import make_splprep


def encontrar_maximo_cuadratica(arr_x, arr_y=None):
    """
    Encontrar el máximo de un array 1d que tiene forma de cuadrática
    ajustando un polinomio de orden 2
    """
    if arr_y is None:
        arr_y = arr_x
        arr_x = np.arange(len(arr_y))

    # Ajustar el polinomio de orden 2
    coeffs = np.polyfit(arr_x, arr_y, 2)
    poly = np.poly1d(coeffs)

    # Encontrar el máximo del polinomio
    a, b, c = coeffs
    max_x = -b / (2 * a)
    max_y = poly(max_x)

    return max_x, max_y


def spline_zeros(rc, z):
    """
    Find a smooth spline curve that passes through the points (r, c) that are zero, where z >= 0.
    ETERNO
    """
    new_order = np.argsort(rc[:, 0])
    xy = np.fliplr(rc[new_order])
    z = z[new_order]
    w = 1 / z
    spline, u = make_splprep(xy.T, w=w, s=z.size)
    xy_out = np.zeros((xy[-1, 1] - xy[0, 1] + 1, 2))
    xy_out[:, 1] = np.arange(xy[0, 1], xy[-1, 1] + 1)
    for i, y in enumerate(xy_out[:, 1]):
        xy_out[i, 0] = spline(y)[0]
    return xy_out
