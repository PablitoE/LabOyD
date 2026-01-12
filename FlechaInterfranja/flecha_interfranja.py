from typing import Tuple
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
from scipy.signal import find_peaks, peak_prominences
from scipy.optimize import minimize, curve_fit, minimize_scalar
from scipy.stats import linregress
from scipy import fftpack
from scipy.ndimage import rotate
from uncertainties import ufloat
from uncertainties.unumpy import nominal_values, sqrt
import os
import re
import logging
from datetime import datetime
from heapq import nlargest, nsmallest
from Varios.optimizations import encontrar_maximo_cuadratica, proportionality_with_uncertainties, OptimizerState
from Varios.lines_points import associate_two_sets_of_lines, rotate_2d_points
from Varios.images import log_normalize


WAVELENGTH_NM = 633

ROTATION_IGNORE_LOW_FREQ_PIXELS = 3
ROTATION_RANGE_ANGLE_DEG = 4
ROTATION_N_RANGE_ANGLE = 13
MINIMUM_DISTANCE_PEAKS = 10
PROMINENCE_PEAKS = 1
GAUSSIAN_BLUR_SIGMA = 2
GAUSSIAN_BLUR_SIGMAY_FACTOR = 2
GAUSSIAN_BLUR_KERNEL_SIZE = 17
GAUSSIAN_BLUR_SIGMA_CIRCLE = 16
GAUSSIAN_BLUR_KERNEL_SIZE_CIRCLE = 51
HOUGH_PARAM1 = 5
HOUGH_PARAM2 = 5
FRACTION_OF_SEPARATION_TO_SEARCH_FRINGES = 0.8
MINIMUM_DISTANCE_FROM_EDGES = 70
DISCARD_EDGE_POINTS = 2
FIND_FRINGES_STEP = 25       # px
FIND_FRINGES_APERTURE_IN_SEARCH = 0.4
MAX_NUMBER_POINTS_FIT = 7
REQUIRED_IMS = 10
RESULTS_DIR = "results"
IQR_FACTOR_IMS = 1.0
IQR_FACTOR_POINTS_IN_FRINGES = 3.5
OPTIMIZE_REGULARIZER_MAX_DEV = 0.15
UNCERTAINTY_MAX_DISTANCE_PX = 5
N_LARGEST_DISTANCES_TO_VALLEY_CURVES = 5
N_LARGEST_DISTANCES_FOR_ARROW = 3

SHOW_ALL = False
SHOW_EACH_RESULT = False
SAVE_RESULTS = True
PLOT_CIRCLES_DEBUG = False
TRACK_OPTIMIZATION = False

# Ruta de tu imagen o directorio con imágenes que comienzan con un número
image_path = r"/home/pablo/OneDrive/Documentos/INTI-Calibraciones/Planos/LMD 2025/Imágenes/TOP"  # noqa: E501
output_dir = os.path.join(image_path, RESULTS_DIR)
os.makedirs(output_dir, exist_ok=True)
logger = logging.getLogger(__name__)


def find_equidistant_peaks(signal, max_min='max', **kwargs):
    signal = signal if max_min == 'max' else -signal

    prominence = kwargs.pop('prominence', None)
    peaks, properties = find_peaks(signal, prominence=prominence, **kwargs)
    if len(peaks) < 3:  # Not enough peaks to keep equidistant ones
        return peaks
    d_peaks = np.diff(peaks)
    good_differences = eliminar_outliers_iqr(d_peaks, return_mask=True, iqr_factor=1)
    if np.all(good_differences):    # No suspicious peaks
        return peaks

    # Look for the most likely location of peaks
    median_d_peak = np.median(d_peaks)

    def model_shift_l1(x, period, points):
        closest_n = np.round((points - x) / period).astype(int)
        return np.sum(np.abs(points - closest_n * period - x))

    res = minimize_scalar(model_shift_l1, bounds=(-median_d_peak / 2, median_d_peak / 2), args=(median_d_peak, peaks))
    closest_ns = np.round((peaks - res.x) / median_d_peak).astype(int)

    # Remove extra peaks by keeping the one closest to the median, and add peaks where missing
    prominences = peak_prominences(signal, peaks)[0] if prominence is None else properties['prominences']
    min_n = np.min(closest_ns)
    max_n = np.max(closest_ns)
    for n in range(min_n, max_n + 1):
        idx_peaks_at_n = np.where(closest_ns == n)[0]
        if len(idx_peaks_at_n) > 1:
            closest_peaks = peaks[idx_peaks_at_n]
            idx_peak_at_n = np.argmin(np.abs(closest_peaks - (n * median_d_peak + res.x)) / prominences[idx_peaks_at_n])
            idx_peaks_at_n = np.delete(idx_peaks_at_n, idx_peak_at_n)
            peaks = np.delete(peaks, idx_peaks_at_n)
        if n not in closest_ns:
            peaks = np.insert(peaks, n - min_n, n * median_d_peak + res.x)

    return peaks


def scaledLinearOp_To_array(scaledLinearOp):
    identity = np.eye(scaledLinearOp.shape[0])
    return scaledLinearOp.matmat(identity)


def rotate_image_to_max_frequency(
    img_array, ignore_low_freq_pixels=2, precision=3, range_angle_deg=4, n_range_angle=11, n_refine_neighbors=2,
    n_refine_add_between=2, central_rows_ratio=None
):
    Nr, Nc = img_array.shape
    Nr_ = Nr * precision
    Nc_ = Nc * precision
    ignore_low_freq_pixels *= precision
    # Calcular la transformada de Fourier 2D
    fft_img = fftpack.fftshift(fftpack.fft2(img_array, shape=(Nr_, Nc_)))

    # Ignorar la componente continua y una cierta cantidad de pixeles aledaños
    fft_img[Nr_//2-ignore_low_freq_pixels:Nr_//2+ignore_low_freq_pixels,
            Nc_//2-ignore_low_freq_pixels:Nc_//2+ignore_low_freq_pixels] = 0

    # Encontrar el máximo de frecuencia espacial en el primer o cuarto cuadrante
    fft_img = fft_img[:, Nc_//2:]
    max_freq_idx = np.unravel_index(np.argmax(np.abs(fft_img)), fft_img.shape)
    max_freq_idx = (Nr_//2 - max_freq_idx[0], max_freq_idx[1])

    # Calcular el ángulo de rotación
    angle = - np.rad2deg(np.arctan2(max_freq_idx[0], max_freq_idx[1]))

    # Definir filas centrales para el análisis
    if central_rows_ratio is None:
        central_rows_ratio = 1
    assert 0 < central_rows_ratio <= 1, "central_rows_ratio debe ser menor o igual a 1"
    central_rows = slice(int(Nr * (0.5 - central_rows_ratio / 2)), int(Nr * (0.5 + central_rows_ratio / 2)))

    # Proponer varios ángulos posibles
    variations_cum = np.zeros(n_range_angle)
    possible_angles = np.linspace(angle - range_angle_deg, angle + range_angle_deg,
                                  n_range_angle)
    cumulative_intensity = np.sum(img_array[central_rows].astype(np.float64), axis=0)
    if Nc % 2 == 0:
        cumulative_intensity = cumulative_intensity[Nc//2:] + np.flip(cumulative_intensity[:Nc//2])
    else:
        cumulative_intensity = cumulative_intensity[(Nc//2)+1:] + np.flip(cumulative_intensity[:Nc//2])
    cumulative_intensity /= np.sum(cumulative_intensity)
    cumulative_intensity = np.cumsum(cumulative_intensity)
    limit = np.where(cumulative_intensity > 0.95)[0][0]

    for ka, angle in enumerate(possible_angles):
        rotated_img = rotate(img_array, angle, mode='nearest', reshape=False)
        cumulative_intensity = np.sum(rotated_img[central_rows], axis=0)
        variations_cum[ka] = np.var(cumulative_intensity[Nc // 2 - limit:Nc // 2 + limit])

    # Proponer ángulos posibles cercanos al máximo
    idx_max = np.argmax(variations_cum)
    start_refine = max(0, idx_max - n_refine_neighbors)
    end_refine = min(n_range_angle, idx_max + n_refine_neighbors + 1)
    new_possible_angles = np.linspace(
        possible_angles[start_refine], possible_angles[end_refine - 1],
        (n_refine_add_between + 1) * (end_refine - start_refine) - n_refine_add_between
    )
    new_variations_cum = np.zeros(len(new_possible_angles))
    for ka, angle in enumerate(new_possible_angles):
        if angle in possible_angles:
            new_variations_cum[ka] = variations_cum[np.where(possible_angles == angle)[0][0]]
            continue
        rotated_img = rotate(img_array, angle, mode='nearest', reshape=False)
        cumulative_intensity = np.sum(rotated_img[central_rows], axis=0)
        new_variations_cum[ka] = np.var(cumulative_intensity[Nc // 2 - limit:Nc // 2 + limit])
    possible_angles = np.concatenate(
        (possible_angles[:start_refine], new_possible_angles, possible_angles[end_refine:])
    )
    variations_cum = np.concatenate((
        variations_cum[:start_refine], new_variations_cum, variations_cum[end_refine:]
    ))

    angle, _, _ = encontrar_maximo_cuadratica(possible_angles, variations_cum, show=False)

    if angle < possible_angles[0] or angle > possible_angles[-1]:
        angle = possible_angles[np.argmax(variations_cum)]

    # Rotar la imagen
    rotated_img = rotate(img_array, angle, mode='nearest', reshape=False)

    return rotated_img, angle


def blurrear_imagen(img, kernel_size=GAUSSIAN_BLUR_KERNEL_SIZE,
                    sigma=GAUSSIAN_BLUR_SIGMA, sigmaY_factor=0):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma, sigmaY=sigmaY_factor * sigma)


def enmascarar_imagen(img, center_x, center_y, radius):
    mask = np.ones_like(img, dtype=np.uint8) * 255
    cv2.circle(mask, (center_x, center_y), radius, 0, -1)
    if img.dtype == np.uint8:
        return cv2.bitwise_or(img, mask)
    elif img.dtype in (np.float32, np.float64):
        img[mask == 255] = 255
        return img
    else:
        raise ValueError("El tipo de la imagen debe ser uint8 o float32.")


def search_points_in_valley(img, x, y, step=1, aperture=1, discard_last_points=0, r_squared_threshold=0.999):
    edge_value = img[0, 0]
    array_aperture = np.arange(-aperture, aperture + 1)
    x_prev, y_prev = x, y
    new_points = []
    clean = True
    # save_for_debug_detected = False
    while True:
        new_min_y = y_prev - step
        inspect_values = img[new_min_y, x_prev + array_aperture]
        if np.any(inspect_values == edge_value):
            break

        new_min_x, value_y, r_squared = encontrar_maximo_cuadratica(
            x_prev + array_aperture, inspect_values, extreme="min", max_number_points=MAX_NUMBER_POINTS_FIT
        )
        condition = (
            not np.isnan(new_min_x)
            and r_squared > r_squared_threshold
            and (x_prev - aperture <= new_min_x <= x_prev + aperture)
        )
        if condition:
            new_points.append((new_min_x, new_min_y))
            if not clean:
                clean = True
                """
                # Debugging
                save_for_debug_detected = True
                with open("debug_interrupted_fringe_search.pkl", "wb") as f:
                    data_to_save = {
                        "img": img, "x": x, "y": y, "step": step, "aperture": aperture,
                        "discard_last_points": discard_last_points, "r_squared_threshold": r_squared_threshold
                    }
                    pickle.dump(data_to_save, f)
                # quit()
                """
        else:
            clean = False

        """
        # Debugging
        if save_for_debug_detected:
            just_min_in_inspect = np.argmin(inspect_values)
            just_min = just_min_in_inspect + x_prev - aperture
            plt.plot(x_prev + array_aperture, inspect_values)
            plt.plot(just_min, inspect_values[just_min_in_inspect], 'ro', label='Mínimo simple')
            plt.plot(new_min_x, value_y, 'go', label='Mínimo cuadrático. R²={:.3f}'.format(r_squared))
            plt.legend(loc='upper right')
            # plt.show(block=False)
            # plt.pause(0.25) if condition else plt.pause(1)
            # plt.cla()
            plt.show()
            quit()
        """

        if condition:
            x_prev = int(np.round(new_min_x))
        y_prev = new_min_y

    new_points = np.array(new_points)
    if new_points.shape[0] > discard_last_points:
        new_points = new_points[:-discard_last_points]
    return np.reshape(new_points, (-1, 2))


def distances_sets_of_points_to_lines(sets_of_points, slope, intercepts, signed=False):
    "Como las rectas son normalmente verticales, conviene usar un modelo: x = my + b"
    slope_proportion = np.sqrt(1 + slope ** 2)
    distances = []
    for points, b in zip(sets_of_points, intercepts):
        d = (slope * points[:, 1] - points[:, 0] + b) / slope_proportion
        if not signed:
            d = np.abs(d)
        distances.append(d)
    return distances


def maximum_estimator_uniform(values, uncertainty_mode="std", confidence=0.95):
    "Gizapedia: english_beamer_uniform.pdf"
    n = len(values)
    max_estimation = (n + 1) / n * np.max(values)
    if uncertainty_mode == "std":
        u_estimation = np.std(values) / np.sqrt(n)
    elif uncertainty_mode == "confidence":
        u_estimation = confidence ** (1 / n) * max_estimation - max_estimation
    return ufloat(max_estimation, u_estimation)


def optimize_lines(fringes, regularizer_max_dev=0, track=False, n_largest_arrows=3):
    "Como las rectas son normalmente verticales, conviene usar un modelo: x = my + b"
    def mse(parameters, regularized_max_dev, fringes):
        slope = parameters[0]
        interfringe_y = parameters[1]
        first_intercept = parameters[2]
        n_fringes = len(fringes)
        intercepts = first_intercept + np.arange(n_fringes) * interfringe_y
        total_se = 0
        all_distances = distances_sets_of_points_to_lines(fringes, slope, intercepts, signed=True)
        for distances in all_distances:
            total_se += np.sum(distances ** 2)
        total_points = np.sum([len(fringe) for fringe in fringes])
        loss = total_se / total_points
        if regularized_max_dev > 0:
            arrows = np.array([np.max(distances) - np.min(distances) for distances in all_distances])
            max_arrows = np.asarray(nlargest(n_largest_arrows, arrows))
            penalty = regularized_max_dev * np.mean(max_arrows**2)
        else:
            penalty = 0
        return loss, penalty

    state = OptimizerState(mse, (fringes, ), regularization_parameter=0.0, track_optimization=track,
                           parameter_names=['slope', 'interfringe_y', 'first_intercept'])

    initial_guess = np.zeros(3)
    bounds = [(None, None), (0, 10000), (0, 10000)]
    result = minimize(
        state.objective, initial_guess, method='L-BFGS-B', bounds=bounds, callback=state
    )
    state.reg_lambda = regularizer_max_dev
    result = minimize(
        state.objective, result.x, method='L-BFGS-B', bounds=bounds, callback=state
    )
    state.plot_history()

    slope = result.x[0]
    interfringe_y = result.x[1]
    first_intercept = result.x[2]
    n_fringes = len(fringes)
    intercepts = first_intercept + np.arange(n_fringes) * interfringe_y

    """
    # Debugging plot
    if result.fun > 10:
        plt.cla()
        colors = plt.cm.rainbow(np.linspace(0, 1, len(fringes)))
        for i_f, f in enumerate(fringes):
            plt.plot(f[:, 1], f[:, 0], 'o', color=colors[i_f])
            x_min = np.min(f[:, 1])
            x_max = np.max(f[:, 1])
            plt.plot(
                [x_min, x_max], [intercepts[i_f] + slope * x_min, intercepts[i_f] + slope * x_max], color=colors[i_f]
            )
        plt.title('Ajuste de lineas a franjas detectadas. R²={:.3f}'.format(result.fun))
        plt.show(block=False)
        plt.pause(0.2)
    """

    total_points = np.sum([len(fringe) for fringe in fringes])
    rms = result.hess_inv * result.fun / (total_points - len(result.x))
    if not isinstance(rms, np.ndarray):
        rms = scaledLinearOp_To_array(rms)
    interfringe_y = ufloat(interfringe_y, np.sqrt(rms[1, 1]))
    slope_u = ufloat(slope, np.sqrt(rms[0, 0]))
    interfringe = 1 / sqrt(1 + slope_u**2) * interfringe_y
    return slope, intercepts, interfringe


def remove_gaussian_from_curve(data):
    def gaussiana(x, a, x0, sigma):
        return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

    x_ = np.arange(len(data))
    popt, _ = curve_fit(gaussiana, x_, data)
    return data - gaussiana(x_, *popt)


def obtener_gaussiana_2D(img):
    def gaussiana_2d(xy, a, x0, y0, sigma_x, sigma_y):
        x = xy[0]
        y = xy[1]
        return a * np.exp(
            -((x - x0) ** 2 / (2 * sigma_x**2) + (y - y0) ** 2 / (2 * sigma_y**2))
        )

    # Crear un arreglo 2D de coordenadas x e y
    x = np.arange(img.shape[1])
    y = np.arange(img.shape[0])
    X, Y = np.meshgrid(x, y)

    # Ajustar la curva con la gaussiana 2D
    p0 = np.array([1, img.shape[1]/2, img.shape[0]/2, 10, 10])
    XY = np.vstack((X.ravel(), Y.ravel()))
    popt, _ = curve_fit(gaussiana_2d, XY, img.ravel(), p0=p0, xtol=1e-6, ftol=1e-6, maxfev=1000)
    z_values = gaussiana_2d(XY, *popt)
    return z_values.reshape(img.shape)


def eliminar_outliers_iqr(data, return_mask=False, iqr_factor=1.5, only_upper=False):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - iqr_factor * iqr
    upper_bound = q3 + iqr_factor * iqr
    if only_upper:
        mask = data <= upper_bound
    else:
        mask = (data >= lower_bound) & (data <= upper_bound)
    if return_mask:
        return mask
    return data[mask]


def analyze_dir_or_image(image_path, reutilize_saved_results=True):
    if os.path.isdir(image_path):
        output_dir = os.path.join(image_path, RESULTS_DIR)
        if SAVE_RESULTS:
            os.makedirs(output_dir, exist_ok=True)
        save_mid_results_path = os.path.join(output_dir, "flechas_vs_interfranjas.npz")
        if reutilize_saved_results and os.path.isfile(save_mid_results_path) and input(
            "Los resultados ya fueron guardados. ¿Deseas utilizarlos? [Y/n]"
        ) != "n":
            archivo = np.load(save_mid_results_path)
            flechas = archivo["flechas"]
            interfranjas = archivo["interfranjas"]
            image_files = archivo["image_files"]
        else:
            flechas = []
            interfranjas = []
            image_files = []
            for image_file in os.listdir(image_path):
                image_file_path = os.path.join(image_path, image_file)
                if (
                    re.match(r'^\d+', image_file) is not None
                    or re.match(r'^[\w\-\s\.]+?\d+\.[A-Za-z0-9]+$', image_file) is not None
                ) and cv2.haveImageReader(image_file_path):
                    logging.info("Analizando %s", image_file)

                    i, f = analyze_interference(image_file_path)

                    flechas.append(f)
                    interfranjas.append(i)
                    image_files.append(image_file)
            flechas = nominal_values(flechas)
            interfranjas = nominal_values(interfranjas)
            image_files = np.array(image_files)

            if len(flechas) < REQUIRED_IMS:
                logging.info(
                    "No se encontraron suficientes imagenes para el análisis de acuerdo "
                    "al procedimiento (%s imágenes).", REQUIRED_IMS
                )

            if SAVE_RESULTS:
                np.savez_compressed(save_mid_results_path, flechas=flechas,
                                    interfranjas=interfranjas, image_files=image_files)

        save_path = (
            os.path.join(output_dir, "flechas_vs_interfranjas.svg")
            if SAVE_RESULTS else None
        )
        plot_flechas_interfranjas(
            flechas, interfranjas, save_path=save_path, iqr_factor=IQR_FACTOR_IMS
        )

    else:
        analyze_interference(image_path)


def plot_rotation(img, img_rotada, angle):
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    axs[0].title.set_text("Imagen Original")
    axs[0].imshow(img, cmap='gray')
    axs[1].title.set_text("Imagen Rotada {}°".format(angle))
    im = axs[1].imshow(img_rotada, cmap='gray')
    fig.colorbar(im, ax=axs[1])
    plt.show()


def plot_minima_profile(profile, minima_indices):
    plt.figure()
    plt.title("Detección de Mínimos en el Perfil de Intensidad")
    plt.plot(profile, label='Perfil de intensidad')
    plt.plot(minima_indices, profile[minima_indices], 'ro', label='Mínimos detectados')
    plt.xlabel('Posición (píxeles)')
    plt.ylabel('Intensidad promedio')
    plt.legend()
    plt.show()


def analyze_interference(image_path=None, image_array=None, show=SHOW_ALL,
                         show_result=SHOW_EACH_RESULT, save=SAVE_RESULTS,
                         debugging_info=None, regularizer_parameter=OPTIMIZE_REGULARIZER_MAX_DEV,
                         n_largest_distances_for_arrow=N_LARGEST_DISTANCES_FOR_ARROW
                         ) -> Tuple[ufloat, ufloat]:
    assert image_path is not None or image_array is not None
    date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if image_array is None:
        # Cargar la imagen en escala de grises
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("No se pudo cargar la imagen. Verifica la ruta del archivo.")
    else:
        img = image_array

    # Rotar la imagen para dejar las franjas más o menos verticales
    img_rotada, angle_rotated = rotate_image_to_max_frequency(img,
                                                              ignore_low_freq_pixels=ROTATION_IGNORE_LOW_FREQ_PIXELS,
                                                              range_angle_deg=ROTATION_RANGE_ANGLE_DEG,
                                                              n_range_angle=ROTATION_N_RANGE_ANGLE,
                                                              n_refine_neighbors=2,
                                                              n_refine_add_between=1,
                                                              central_rows_ratio=0.2)
    if np.all(img_rotada == 0):
        print("Todo cero!")
        with open(f"{date}_debug_failed_rotation.pkl", "wb") as f:
            data_to_save = {
                "img": img,
                "ignore_low_freq_pixels": ROTATION_IGNORE_LOW_FREQ_PIXELS
            }
            pickle.dump(data_to_save, f)
        raise ValueError("La imagen rotada quedó toda en cero.")
    if debugging_info is not None:
        debugging_info["rotation_angle_estimated"] = angle_rotated

    # Mostrar la imagen original y la rotada
    if show:
        plot_rotation(img, img_rotada, angle_rotated)

    # Detectar franjas de interferometría
    # Sumar la intensidad a lo largo de las filas para proyectar las franjas
    blurred_for_analysis = blurrear_imagen(img_rotada, GAUSSIAN_BLUR_KERNEL_SIZE,
                                           GAUSSIAN_BLUR_SIGMA, GAUSSIAN_BLUR_SIGMAY_FACTOR)
    intensity_profile = np.mean(blurred_for_analysis, axis=0)
    intensity_profile = remove_gaussian_from_curve(intensity_profile)

    # Encontrar los mínimos en el perfil de intensidad
    peaks = find_equidistant_peaks(intensity_profile, max_min='min', prominence=PROMINENCE_PEAKS,
                                   distance=MINIMUM_DISTANCE_PEAKS)

    # Ploteo de los mínimos detectados en el perfil de intensidad
    if show:
        plot_minima_profile(intensity_profile, peaks)

    # Detectar el círculo principal que contiene las franjas
    # Aplicar un desenfoque para reducir ruido
    blurred = blurrear_imagen(img_rotada, GAUSSIAN_BLUR_KERNEL_SIZE_CIRCLE,
                              GAUSSIAN_BLUR_SIGMA_CIRCLE)

    # Usar HoughCircles para detectar el círculo
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.8, minDist=400,
                               param1=HOUGH_PARAM1, param2=HOUGH_PARAM2, minRadius=400,
                               maxRadius=0)

    if circles is not None:
        if PLOT_CIRCLES_DEBUG:
            plt.figure(figsize=(8, 6))
            plt.title("Círculos Detectado")
            for circle in circles[0, :]:
                center_x, center_y, radius = np.uint16(np.around(circle))
                cv2.circle(blurred, (center_x, center_y), radius, (255, 0, 0), 2)
                cv2.circle(blurred, (center_x, center_y), 2, (255, 0, 0), 3)
            plt.imshow(blurred, cmap='gray')
            plt.show()

        circles = np.round(circles[0, :]).astype("int")
        # Tomar el primer círculo detectado
        circle = circles[0]
        center_x, center_y, radius = circle
        # Ajustar el radio del círculo que fue agrandado por el blur
        radius = radius - GAUSSIAN_BLUR_KERNEL_SIZE

        # Mostrar el círculo detectado
        output = blurred.copy()
        cv2.circle(output, (center_x, center_y), radius, (255, 0, 0), 2)
        cv2.circle(output, (center_x, center_y), 2, (255, 0, 0), 3)

        if show:
            plt.figure(figsize=(8, 6))
            plt.title("Círculo Detectado")
            plt.imshow(output, cmap='gray')
            plt.show()
    else:
        with open(f"{date}_debug_no_circles_found.pkl", "wb") as f:
            data_to_save = {
                "image": blurred, "method": cv2.HOUGH_GRADIENT, "dp": 1.8, "minDist": 400, "param1": HOUGH_PARAM1,
                "param2": HOUGH_PARAM2, "minRadius": 400, "maxRadius": 0
            }
            pickle.dump(data_to_save, f)
        raise ValueError("No se detectó ningún círculo en la imagen.")

    # Descartar franjas cercanas al borde del círculo
    circle_left = center_x - radius + MINIMUM_DISTANCE_FROM_EDGES
    circle_right = center_x + radius - MINIMUM_DISTANCE_FROM_EDGES
    peaks = [peak for peak in peaks if circle_left < peak < circle_right]

    # Analizar cada franja en una imagen blurreada, normalizada y enmascarada
    blurred_for_analysis = log_normalize(blurred_for_analysis, GAUSSIAN_BLUR_SIGMA_CIRCLE)
    blur_masked = enmascarar_imagen(blurred_for_analysis, center_x, center_y, radius)

    if show:
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        axs[0].imshow(blurred_for_analysis, cmap='gray')
        axs[0].set_title("Imagen Blurreada")
        axs[1].imshow(blurred, cmap='gray')
        axs[1].set_title("Imagen Blurreada para Círculo")
        axs[2].imshow(blur_masked, cmap='gray')
        axs[2].set_title("Imagen Blurreada y Normalizada")
        plt.show()

    # Encontrar puntos de cada franja
    x_positions, y_positions = [], []
    avg_separation = np.mean(np.diff(peaks))
    if np.isnan(avg_separation):
        plot_rotation(img, img_rotada, angle_rotated)
        plot_minima_profile(intensity_profile, peaks)
        img_rotada, angle_rotated = rotate_image_to_max_frequency(
            img, ignore_low_freq_pixels=ROTATION_IGNORE_LOW_FREQ_PIXELS
        )  # debugging
        raise ValueError("No se encontraron franjas en la imagen.")
    n_search_fringe = int(avg_separation * FRACTION_OF_SEPARATION_TO_SEARCH_FRINGES) // 2
    y_range = np.arange(
        int(center_y - n_search_fringe), int(center_y + n_search_fringe + 1)
    )
    for peak in peaks:
        x_range = np.arange(
            int(peak - n_search_fringe), int(peak + n_search_fringe + 1)
        )
        peak_area = blur_masked[np.ix_(y_range, x_range)]
        local_min = np.unravel_index(np.argmin(peak_area), peak_area.shape)
        x_positions.append(local_min[1] + x_range[0])
        y_positions.append(local_min[0] + y_range[0])

        # plt.imshow(peak_area, cmap='gray')
        # plt.plot(local_min[1], local_min[0], 'ro')
        # plt.plot(n_search_fringe, n_search_fringe, 'bo')
        # plt.show()

    x_positions = np.array(x_positions)
    y_positions = np.array(y_positions)

    # Detección de puntos sospechosos
    difs = np.diff(x_positions)
    avg_dif = np.median(difs)
    inliers = np.abs(difs - avg_dif) < n_search_fringe
    inliers = np.insert(inliers, 0, True)

    # Ploteo de los mínimos detectados en la imagen original
    if show:
        plt.figure(figsize=(10, 6))
        plt.title("Mínimos Detectados en la Imagen")
        plt.imshow(blur_masked, cmap='gray')
        plt.plot(x_positions[inliers], y_positions[inliers], 'ro',
                 label='Mínimos detectados')
        plt.plot(x_positions[~inliers], y_positions[~inliers], 'rx',
                 label='Mínimos descartados')
        plt.legend()
        plt.show()

    x_positions = x_positions[inliers]
    y_positions = y_positions[inliers]

    # Encontrar franjas de interferometría
    n_fringes = len(x_positions)
    fringes = [None] * n_fringes
    find_fringes_aperture = int(n_search_fringe * FIND_FRINGES_APERTURE_IN_SEARCH)
    logging.info(
        "Encontradas %d franjas. Buscando puntos cada %d filas, en rango de %d pixeles.",
        n_fringes, FIND_FRINGES_STEP, find_fringes_aperture,
    )
    for i, (x, y) in enumerate(zip(x_positions, y_positions)):
        # Encontrar los puntos más oscuros dentro de la franja subiendo y bajando
        upper_points = search_points_in_valley(blur_masked, x, y, FIND_FRINGES_STEP,
                                               find_fringes_aperture, discard_last_points=DISCARD_EDGE_POINTS)
        lower_points = search_points_in_valley(blur_masked, x, y, -FIND_FRINGES_STEP,
                                               find_fringes_aperture, discard_last_points=DISCARD_EDGE_POINTS)

        # Unir los puntos de arriba y abajo en una sola franja
        fringes[i] = np.concatenate((upper_points[::-1], [(x, y)], lower_points))

    # Encontrar lineas ideales correspondientes a las franjas
    rotated_valley_curves = None
    if debugging_info is not None and "valley_curves" in debugging_info.keys():
        rotated_valley_curves = rotate_2d_points(debugging_info["valley_curves"], -angle_rotated, shape=img.shape)
        # Valley curves are given as (row, column) but fringes are (x, y)
        fringe_index_in_valley_curves, distances_fringes_to_valley = associate_two_sets_of_lines(
            rotated_valley_curves, fringes, flip=True
        )

    # Ajustar franjas con rectas
    slope, intercepts, interfringe_distance = optimize_lines(
        fringes, regularizer_max_dev=regularizer_parameter, track=TRACK_OPTIMIZATION
    )

    # Actualizar valor de rotación estimada
    angle_rotated = angle_rotated - np.rad2deg(np.arctan(slope))
    if debugging_info is not None:
        debugging_info["rotation_angle_estimated_corrected"] = angle_rotated

    logging.info("Distancias media entre rectas: %s (k=1)", interfringe_distance)

    # Calcular desviación máxima de la recta
    all_distances = distances_sets_of_points_to_lines(
        fringes, slope, intercepts, signed=True
    )
    max_distance_positive = np.zeros(len(all_distances), dtype=[('index', int), ('value', float)])
    max_distance_negative = np.zeros(len(all_distances), dtype=[('index', int), ('value', float)])
    rms_n_max_distances = np.zeros(len(all_distances))

    if debugging_info is not None and "valley_curves" in debugging_info.keys():
        rmsd_to_valley_curves = np.zeros(len(all_distances))
        largest_distances_to_valley_curves = np.array([])
    mask_outliers = []
    for i, distances in enumerate(all_distances):
        mask = eliminar_outliers_iqr(
            distances, return_mask=True, iqr_factor=IQR_FACTOR_POINTS_IN_FRINGES
        )
        distances = distances[mask]
        mask_outliers.append(mask)
        if n_largest_distances_for_arrow > 1:
            n_max_distances_positive = np.asarray(nlargest(n_largest_distances_for_arrow, distances))
            n_max_distances_negative = np.asarray(nsmallest(n_largest_distances_for_arrow, distances))
            distances_matrix = n_max_distances_positive[:, np.newaxis] - n_max_distances_negative[np.newaxis, :]
            distances_matrix_upper_triangle = distances_matrix[np.triu_indices_from(distances_matrix, k=0)]
            rms_n_max_distances[i] = np.sqrt(np.mean(distances_matrix_upper_triangle**2))

        max_distance_positive[i]['index'] = np.argmax(distances)
        max_distance_negative[i]['index'] = np.argmin(distances)
        max_distance_positive[i]['value'] = distances[max_distance_positive[i]['index']]
        max_distance_negative[i]['value'] = distances[max_distance_negative[i]['index']]
        if debugging_info is not None and "valley_curves" in debugging_info.keys():
            diffs = distances_fringes_to_valley[i][mask]
            rmsd_to_valley_curves[i] = np.sqrt(np.mean(diffs ** 2))
            largest_distances_to_valley_curves = np.hstack(
                (largest_distances_to_valley_curves, np.asarray(nlargest(N_LARGEST_DISTANCES_TO_VALLEY_CURVES, diffs)))
            )
    total_distances = max_distance_positive['value'] - max_distance_negative['value']
    if n_largest_distances_for_arrow == 1:
        arrow_value = np.max(total_distances).astype(float)
    else:
        arrow_value = np.max(rms_n_max_distances).astype(float)
    fringe_with_max_deviation = np.argmax(total_distances)
    error_max_distance = UNCERTAINTY_MAX_DISTANCE_PX  # np.std(total_distances) / np.sqrt(len(total_distances))
    arrow = ufloat(arrow_value, error_max_distance)
    logging.info("Desviación máxima de la recta: %s (k=1)", arrow)
    if debugging_info is not None and "valley_curves" in debugging_info.keys():
        debugging_info["rmsd_to_valley_curves"] = np.mean(rmsd_to_valley_curves)
        debugging_info["avg_largest_distances_to_valley_curves"] = np.mean(
            nlargest(N_LARGEST_DISTANCES_TO_VALLEY_CURVES, largest_distances_to_valley_curves)
        )
        debugging_info["max_distance_fringe_to_valley_curve"] = np.max(largest_distances_to_valley_curves)

    # Ploteo de los mínimos detectados en la imagen original
    if save or show_result:
        colores = ['r', 'g', 'b']
        plt.figure(figsize=(12, 8))
        plt.title("Franjas detectadas en la Imagen")
        plt.imshow(blur_masked, cmap='gray')
        for i, fringe in enumerate(fringes):
            good_ones = mask_outliers[i]
            good_ones_dots = fringe[good_ones]
            plt.plot(good_ones_dots[:, 0], good_ones_dots[:, 1], 'o',
                     color=colores[i % len(colores)], label=f'Franja #{i}')
            marker_max = '*' if i == fringe_with_max_deviation else 'o'
            plt.plot(good_ones_dots[max_distance_positive[i]['index'], 0],
                     good_ones_dots[max_distance_positive[i]['index'], 1], marker_max, color=colores[i % len(colores)],
                     markersize=10)
            plt.plot(good_ones_dots[max_distance_negative[i]['index'], 0],
                     good_ones_dots[max_distance_negative[i]['index'], 1], marker_max, color=colores[i % len(colores)],
                     markersize=10)
            plt.plot(fringe[~good_ones, 0], fringe[~good_ones, 1], 'x',
                     color=colores[i % len(colores)])
            x_fit = slope * fringe[:, 1] + intercepts[i]
            plt.plot(x_fit, fringe[:, 1], color=colores[i % len(colores)], linestyle='--')
            if debugging_info is not None and "valley_curves" in debugging_info.keys():
                valley_curve = rotated_valley_curves[fringe_index_in_valley_curves[i]]
                plt.plot(valley_curve[:, 1], valley_curve[:, 0], color=colores[i % len(colores)],
                         linestyle=':', linewidth=1.5)
        plt.legend()
    if save:
        output_dir = os.path.join(os.path.dirname(image_path), RESULTS_DIR)
        os.makedirs(output_dir, exist_ok=True)
        file_name = os.path.splitext(os.path.basename(image_path))[0]
        plt.savefig(os.path.join(output_dir, f"{file_name}.svg"))
    if show_result:
        plt.show()
    if save or show_result:
        plt.close()

    return interfringe_distance, arrow


def analisis_flechas_interfranjas(flechas, interfranjas, iqr_factor=0.75, full_output=True, uncertainties=None):
    if uncertainties is not None and len(uncertainties) == 2:
        pendiente, std_err = proportionality_with_uncertainties(
            interfranjas, flechas, uncertainties[0], uncertainties[1]
        )
        y = pendiente * interfranjas
        intercepto = 0.0
    else:
        pendiente, intercepto, _, _, std_err = linregress(interfranjas, flechas)
        y = pendiente * interfranjas + intercepto
    errores = np.abs(flechas - y)
    mask_buenos = eliminar_outliers_iqr(
        errores, return_mask=True, only_upper=True, iqr_factor=iqr_factor
    )

    descartados = np.where(~mask_buenos)[0]
    no_descartados = np.where(mask_buenos)[0]

    if np.any(~mask_buenos):
        if uncertainties is not None and len(uncertainties) == 2:
            uncertainties = [u[no_descartados] if isinstance(u, np.ndarray) else u for u in uncertainties]
            pendiente, std_err = proportionality_with_uncertainties(
                interfranjas[no_descartados], flechas[no_descartados],
                uncertainties[0], uncertainties[1]
            )
            y = pendiente * interfranjas[no_descartados]
        else:
            pendiente, intercepto, _, _, std_err = linregress(
                interfranjas[no_descartados], flechas[no_descartados]
            )
    x = np.linspace(interfranjas.min(), interfranjas.max(), 4)
    y = pendiente * x + intercepto
    pendiente_u = ufloat(pendiente, std_err)
    if full_output:
        return pendiente_u, x, y, descartados, no_descartados
    else:
        return pendiente_u


def plot_flechas_interfranjas(
    flechas, interfranjas, error_flechas=2.0, save_path=None, iqr_factor=0.75
):
    pendiente_u, x, y, descartados, no_descartados = analisis_flechas_interfranjas(
        flechas, interfranjas, iqr_factor=iqr_factor
    )

    plt.figure(figsize=(8, 6))
    plt.title(f"Flechas vs Interfranjas. Pendiente: {pendiente_u} (k=1)")
    plt.errorbar(
        interfranjas[no_descartados], flechas[no_descartados], yerr=error_flechas,
        fmt='s', color='black', capsize=4, elinewidth=1.2, ecolor='gray'
    )
    plt.errorbar(
        interfranjas[descartados], flechas[descartados], yerr=error_flechas,
        fmt='s', color='red', capsize=4, elinewidth=1.2, ecolor='lightcoral'
    )
    plt.plot(x, y, linestyle='--', color='black', linewidth=1.5)
    plt.xlabel("Interfranjas [px]")
    plt.ylabel("Flechas [px]")
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()

    logging.info(
        "La desviación máxima de planitud es %.2f nm.", pendiente_u * WAVELENGTH_NM / 2
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.FileHandler(f'{output_dir}/flecha_interfranja.log'),
                  logging.StreamHandler()]
    )

    # Ejecutar el análisis
    analyze_dir_or_image(image_path)
