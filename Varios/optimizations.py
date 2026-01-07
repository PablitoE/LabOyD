import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_splprep


def encontrar_maximo_cuadratica(arr_x, arr_y=None, max_number_points=7, extreme="max", show=False):
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

    arr_x_original = arr_x.copy()
    arr_y_original = arr_y.copy()

    # Limitar el número de puntos para el ajuste
    if extreme == "max":
        extreme_index = np.argmax(arr_y)
    elif extreme == "min":
        extreme_index = np.argmin(arr_y)
    else:
        max_index = np.argmax(arr_y)
        d_max_from_center = abs(max_index - len(arr_x) // 2)
        min_index = np.argmin(arr_y)
        d_min_from_center = abs(min_index - len(arr_x) // 2)
        extreme_index = max_index if d_max_from_center < d_min_from_center else min_index
    extreme_location = arr_x[extreme_index]
    extreme_value_by_index = arr_y[extreme_index]
    if len(arr_x) > max_number_points:
        center_index = extreme_index
        half_window = max_number_points // 2
        if center_index - half_window < 0:
            center_index = half_window
        if center_index + half_window >= len(arr_x):
            center_index = len(arr_x) - half_window - 1
        start_index = max(0, center_index - half_window)
        end_index = min(len(arr_x), center_index + half_window + 1)
        arr_x = arr_x[start_index:end_index]
        arr_y = arr_y[start_index:end_index]

    if np.all(arr_y == 0):
        return arr_x[len(arr_x) // 2], 0, 0.0

    # Ajustar el polinomio de orden 2
    coeffs = np.polyfit(arr_x, arr_y, 2)
    poly = np.poly1d(coeffs)

    # Encontrar un nivel de incertidumbre en la posición del extremo
    residuals = arr_y - poly(arr_x)
    ss_res = np.sum(residuals**2)
    ss_y = np.linalg.norm(arr_y)**2
    r_squared = 1 - (ss_res / ss_y)
    if not np.isfinite(r_squared):
        r_squared = 0.0

    # Encontrar el extremo del polinomio
    a, b, c = coeffs
    extr_x = -b / (2 * a)
    extr_y = poly(extr_x)

    # Plot para depuración
    if show:
        import matplotlib.pyplot as plt

        x_fit = np.linspace(np.min(arr_x), np.max(arr_x), 100)
        y_fit = poly(x_fit)

        plt.plot(arr_x_original, arr_y_original, 'o', label='Datos')
        plt.plot(x_fit, y_fit, '-', label='Ajuste cuadrático')
        plt.plot(extr_x, extr_y, 'rx', label='Extremo encontrado')
        plt.legend()
        plt.title(f'Ajuste cuadrático (R² = {r_squared:.4f})')
        plt.show()

    # Determinar si es un máximo o mínimo
    if extreme == "max" and a >= 0:
        return extreme_location, extreme_value_by_index, 0.0
    if extreme == "min" and a <= 0:
        return extreme_location, extreme_value_by_index, 0.0

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


def proportionality_with_uncertainties(x, y, ux, uy):
    a = np.sum(x*y / uy**2) / np.sum(x**2 / uy**2)

    for _ in range(20):
        w = 1 / (uy**2 + (a**2) * ux**2)
        a = np.sum(w * x * y) / np.sum(w * x**2)

    ua = np.sqrt(1 / np.sum(w * x**2))
    return a, ua


class OptimizerState:
    def __init__(self, objective_function, args=(), track_cost_function=False, regularization_parameter=0.0):
        self.lambda_history = []
        self.iteration = 0
        self.fun_history = []
        self.reg_lambda = regularization_parameter
        self.track_cost_function = track_cost_function
        self.objective_function = objective_function
        self.args = args
        self.this_iteration_evals = []

    @property
    def reg_lambda(self):
        return self._reg_lambda

    @reg_lambda.setter
    def reg_lambda(self, value):
        self._reg_lambda = value
        self.lambda_history.append([self.iteration, value])

    def objective(self, x):
        """
        The objective function has arguments (x, regularization_parameter, *args)
        """
        loss, penalty = self.objective_function(x, self.reg_lambda, *self.args)
        fun = loss + penalty
        self.this_iteration_evals.append((x, loss, penalty, fun))
        return fun

    def __call__(self, intermediate_result):
        self.iteration += 1
        values_cost_this_iteration = np.array([[loss, p, f] for _, loss, p, f in self.this_iteration_evals])
        values_cost_current = values_cost_this_iteration[values_cost_this_iteration[:, 2] == intermediate_result.fun]

        if self.track_cost_function:
            self.fun_history.append(values_cost_current)
        self.this_iteration_evals = []

    def plot_history(self):
        if self.track_cost_function:
            fun_history = np.concatenate(self.fun_history, axis=0)
            loss_proportion = fun_history[:, 0] / fun_history[:, 2]
            penalty_proportion = fun_history[:, 1] / fun_history[:, 2]
            fig, axs = plt.subplots(1, 2, figsize=(16, 6))
            axs[0].plot(fun_history[:, 2], label='Objective Function Value')
            for lambda_point in self.lambda_history:
                axs[0].axvline(x=lambda_point[0], color='r', linestyle='--', alpha=0.5,
                               label=f'Lambda = {lambda_point[1]:.3f}')
            axs[0].set_xlabel('Iteration')
            axs[0].set_ylabel('Objective Function Value')
            axs[0].set_title('Objective Function History')
            axs[0].legend()
            axs[1].plot(loss_proportion, label='Loss Proportion')
            axs[1].plot(penalty_proportion, label='Penalty Proportion')
            axs[1].set_xlabel('Iteration')
            axs[1].set_ylabel('Proportion')
            axs[1].set_title('Loss and Penalty Proportions')
            axs[1].legend()
            plt.show()
