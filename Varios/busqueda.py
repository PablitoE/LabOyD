import numpy as np


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
