import numpy as np
from scipy.interpolate import make_splprep


def encontrar_maximo_cuadratica(arr_x, arr_y=None, max_number_points=7, extreme="max"):
    """
    Encontrar el extremo de un array 1d que tiene forma de cuadrática
    ajustando un polinomio de orden 2
    Parameters
    ----------
    arr_x : np.ndarray
        Array con las posiciones en x
    arr_y : np.ndarray, optional
        Array con los valores en y. Si es None, se usa arr_x como arr_y
    max_number_points : int, optional
        Número máximo de puntos a usar para el ajuste. Por defecto es 7
    extreme : str, optional
        Tipo de extremo a buscar: "max" para máximo, "min" para mínimo, "both" para lo que esté más cercano al centro.
        Por defecto es "max"
    Returns
    -------
    extr_x : float
        Posición en x del extremo encontrado
    extr_y : float
        Valor en y del extremo encontrado
    r_squared : float
        Coeficiente de determinación del ajuste
    """
    if arr_y is None:
        arr_y = arr_x
        arr_x = np.arange(len(arr_y))

    if np.all(arr_y == 0):
        return arr_x[len(arr_x) // 2], 0, 0.0

    # Limitar el número de puntos para el ajuste
    if len(arr_x) > max_number_points:
        if extreme == "max":
            center_index = np.argmax(arr_y)
        elif extreme == "min":
            center_index = np.argmin(arr_y)
        else:
            max_index = np.argmax(arr_y)
            d_max_from_center = abs(max_index - len(arr_x) // 2)
            min_index = np.argmin(arr_y)
            d_min_from_center = abs(min_index - len(arr_x) // 2)
            center_index = max_index if d_max_from_center < d_min_from_center else min_index
        half_window = max_number_points // 2
        if center_index - half_window < 0:
            center_index = half_window
        if center_index + half_window >= len(arr_x):
            center_index = len(arr_x) - half_window - 1
        start_index = max(0, center_index - half_window)
        end_index = min(len(arr_x), center_index + half_window + 1)
        arr_x = arr_x[start_index:end_index]
        arr_y = arr_y[start_index:end_index]

    # Ajustar el polinomio de orden 2
    coeffs = np.polyfit(arr_x, arr_y, 2)
    poly = np.poly1d(coeffs)

    # Encontrar un nivel de incertidumbre en la posición del extremo
    residuals = arr_y - poly(arr_x)
    ss_res = np.sum(residuals**2)
    ss_y = np.linalg.norm(arr_y)**2
    r_squared = 1 - (ss_res / ss_y)

    # Encontrar el extremo del polinomio
    a, b, c = coeffs
    extr_x = -b / (2 * a)
    extr_y = poly(extr_x)

    # Determinar si es un máximo o mínimo
    if extreme == "max" and a >= 0:
        return center_index, arr_y[center_index], 0.0
    if extreme == "min" and a <= 0:
        return center_index, arr_y[center_index], 0.0

    return extr_x, extr_y, r_squared


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
