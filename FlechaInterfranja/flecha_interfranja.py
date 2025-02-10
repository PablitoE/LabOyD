import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import minimize, curve_fit
from scipy.stats import linregress
from uncertainties import ufloat
from uncertainties.unumpy import nominal_values
import os
import re

MINIMUM_DISTANCE_PEAKS = 10
PROMINENCE_PEAKS = 1
GAUSSIAN_BLUR_SIGMA = 4
GAUSSIAN_BLUR_KERNEL_SIZE = 17
GAUSSIAN_BLUR_SIGMA_CIRCLE = 16
GAUSSIAN_BLUR_KERNEL_SIZE_CIRCLE = 51
HOUGH_PARAM1 = 5
HOUGH_PARAM2 = 5
FRACTION_OF_SEPARATION_TO_SEARCH_FRINGES = 0.8
MINIMUM_DISTANCE_FROM_EDGES = 15
FIND_FRINGES_STEP = 25       # px
FIND_FRINGES_APERTURE = 8   # px
RESULTS_DIR = "results"

SHOW_ALL = False
SHOW_EACH_RESULT = False
SAVE_RESULTS = True


def blurrear_imagen(img, kernel_size=GAUSSIAN_BLUR_KERNEL_SIZE,
                    sigma=GAUSSIAN_BLUR_SIGMA):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)


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


def search_points_in_valley(img, x, y, step=1, aperture=1):
    edge_value = img[0, 0]
    array_aperture = np.arange(-aperture, aperture + 1)
    x_prev, y_prev = x, y
    new_points = []
    while True:
        new_min_y = y_prev - step
        inspect_values = img[new_min_y, x_prev + array_aperture]
        if np.all(inspect_values == edge_value):
            break
        new_min_x = np.argmin(inspect_values) + x_prev - aperture
        new_points.append((new_min_x, new_min_y))
        x_prev, y_prev = new_min_x, new_min_y
    new_points = np.array(new_points)
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


def optimize_lines(fringes):
    "Como las rectas son normalmente verticales, conviene usar un modelo: x = my + b"
    def mse(parameters, fringes):
        slope = parameters[0]
        interfringe = parameters[1]
        first_intercept = parameters[2]
        n_fringes = len(fringes)
        intercepts = first_intercept + np.arange(n_fringes) * interfringe
        total_se = 0
        all_distances = distances_sets_of_points_to_lines(fringes, slope, intercepts)
        for distances in all_distances:
            total_se += np.sum(distances ** 2)
        return total_se

    initial_guess = np.zeros(3)
    result = minimize(mse, initial_guess, args=fringes, method='BFGS')

    slope = result.x[0]
    interfringe = result.x[1]
    first_intercept = result.x[2]
    n_fringes = len(fringes)
    intercepts = first_intercept + np.arange(n_fringes) * interfringe

    total_points = np.sum([len(fringe) for fringe in fringes])
    rms = result.hess_inv * result.fun / (total_points - len(result.x))
    interfringe = ufloat(interfringe, np.sqrt(rms[1, 1]))
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
    popt, _ = curve_fit(gaussiana_2d, XY, img.ravel(), p0=p0)
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


def analyze_dir_or_image(image_path):
    if os.path.isdir(image_path):
        flechas = []
        interfranjas = []
        for image_file in os.listdir(image_path):
            image_file_path = os.path.join(image_path, image_file)
            if re.match(r'^\d+', image_file) is not None and cv2.haveImageReader(
                image_file_path
            ):
                print(f"Analizando {image_file}")
                i, f = analyze_interference(image_file_path)
                flechas.append(f)
                interfranjas.append(i)
        flechas = nominal_values(flechas)
        interfranjas = nominal_values(interfranjas)

        save_path = None
        if SAVE_RESULTS:
            output_dir = os.path.join(image_path, RESULTS_DIR)
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, "flechas_vs_interfranjas.svg")
        plot_flechas_interfranjas(flechas, interfranjas, save_path=save_path)

    else:
        analyze_interference(image_path)


def analyze_interference(image_path):
    # Cargar la imagen en escala de grises
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("No se pudo cargar la imagen. Verifica la ruta del archivo.")

    # Mostrar la imagen original
    if SHOW_ALL:
        plt.figure(figsize=(8, 6))
        plt.title("Imagen Original")
        plt.imshow(img, cmap='gray')
        plt.colorbar()
        plt.show()

    # Detectar franjas de interferometría
    # Sumar la intensidad a lo largo de las filas para proyectar las franjas
    blurred_for_analysis = blurrear_imagen(img, GAUSSIAN_BLUR_KERNEL_SIZE,
                                           GAUSSIAN_BLUR_SIGMA)
    intensity_profile = np.mean(blurred_for_analysis, axis=0)
    intensity_profile = remove_gaussian_from_curve(intensity_profile)

    # Encontrar los mínimos en el perfil de intensidad
    peaks, _ = find_peaks(-intensity_profile, distance=MINIMUM_DISTANCE_PEAKS,
                          prominence=PROMINENCE_PEAKS)

    # Ploteo de los mínimos detectados en el perfil de intensidad
    if SHOW_ALL:
        plt.figure(figsize=(8, 6))
        plt.title("Detección de Mínimos en el Perfil de Intensidad")
        plt.plot(intensity_profile, label='Perfil de intensidad')
        plt.plot(peaks, intensity_profile[peaks], 'ro', label='Mínimos detectados')
        plt.xlabel('Posición (píxeles)')
        plt.ylabel('Intensidad promedio')
        plt.legend()
        plt.show()

    # Detectar el círculo principal que contiene las franjas
    # Aplicar un desenfoque para reducir ruido
    blurred = blurrear_imagen(img, GAUSSIAN_BLUR_KERNEL_SIZE_CIRCLE,
                              GAUSSIAN_BLUR_SIGMA_CIRCLE)

    # Usar HoughCircles para detectar el círculo
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.8, minDist=100,
                               param1=HOUGH_PARAM1, param2=HOUGH_PARAM2, minRadius=100,
                               maxRadius=0)

    if circles is not None:
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

        if SHOW_ALL:
            plt.figure(figsize=(8, 6))
            plt.title("Círculo Detectado")
            plt.imshow(output, cmap='gray')
            plt.show()
    else:
        raise ValueError("No se detectó ningún círculo en la imagen.")

    # Descartar franjas cercanas al borde del círculo
    circle_left = center_x - radius + MINIMUM_DISTANCE_FROM_EDGES
    circle_right = center_x + radius - MINIMUM_DISTANCE_FROM_EDGES
    peaks = [peak for peak in peaks if circle_left < peak < circle_right]

    # Generar un normalizador
    normalizador = obtener_gaussiana_2D(blurred)
    normalizador = enmascarar_imagen(normalizador, center_x, center_y, radius)

    # Analizar cada franja en una imagen blurreada, normalizada y enmascarada
    blur_masked = enmascarar_imagen(blurred_for_analysis, center_x, center_y, radius)
    blur_masked = blur_masked / normalizador

    # Encontrar puntos de cada franja
    x_positions, y_positions = [], []
    avg_separation = np.mean(np.diff(peaks))
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

    x_positions = np.array(x_positions)
    y_positions = np.array(y_positions)

    # Ploteo de los mínimos detectados en la imagen original
    if SHOW_ALL:
        plt.figure(figsize=(10, 6))
        plt.title("Mínimos Detectados en la Imagen")
        plt.imshow(blur_masked, cmap='gray')
        plt.plot(x_positions, y_positions, 'ro', label='Mínimos detectados')
        plt.legend()
        plt.show()

    # Encontrar franjas de interferometría
    n_fringes = len(x_positions)
    print(f"Encontradas {n_fringes} franjas.")
    fringes = [None] * n_fringes
    for i, (x, y) in enumerate(zip(x_positions, y_positions)):
        # Encontrar los puntos más oscuros dentro de la franja subiendo y bajando
        upper_points = search_points_in_valley(blur_masked, x, y, FIND_FRINGES_STEP,
                                               FIND_FRINGES_APERTURE)
        lower_points = search_points_in_valley(blur_masked, x, y, -FIND_FRINGES_STEP,
                                               FIND_FRINGES_APERTURE)

        # Unir los puntos de arriba y abajo en una sola franja
        fringes[i] = np.concatenate((upper_points[::-1], [(x, y)], lower_points))

    # Ajustar franjas con rectas
    slope, intercepts, interfringe_distance = optimize_lines(fringes)

    # Calcular distancias entre rectas
    """
    intercepts_sorted = np.sort(intercepts)
    slope_proportion = np.sqrt(1 + slope ** 2)
    distances = np.abs(np.diff(intercepts_sorted)) / slope_proportion
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)
    error_mean_distance = std_distance / np.sqrt(len(distances))
    interfringe_distance = ufloat(mean_distance, error_mean_distance)
    """
    print("Distancias media entre rectas:", interfringe_distance, " (k=1)")

    # Calcular desviación máxima de la recta
    all_distances = distances_sets_of_points_to_lines(
        fringes, slope, intercepts, signed=True
    )
    max_distance_positive = np.zeros(len(all_distances))
    max_distance_negative = np.zeros(len(all_distances))
    mask_outliers = []
    for i, distances in enumerate(all_distances):
        mask = eliminar_outliers_iqr(distances, return_mask=True, iqr_factor=2.0)
        distances = distances[mask]
        mask_outliers.append(mask)
        max_distance_positive[i] = np.max(distances)
        max_distance_negative[i] = np.min(distances)
    total_distances = max_distance_positive - max_distance_negative
    max_total_distance = np.max(total_distances)
    error_max_distance = np.std(total_distances) / np.sqrt(len(total_distances))
    flecha = ufloat(max_total_distance, error_max_distance)
    print("Desviación máxima de la recta:", flecha, " (k=1)")

    # Ploteo de los mínimos detectados en la imagen original
    colores = ['r', 'g', 'b']
    plt.figure(figsize=(12, 8))
    plt.title("Franjas detectadas en la Imagen")
    plt.imshow(blur_masked, cmap='gray')
    for i, fringe in enumerate(fringes):
        good_ones = mask_outliers[i]
        plt.plot(fringe[good_ones, 0], fringe[good_ones, 1], 'o',
                 color=colores[i % len(colores)], label=f'Franja #{i}')
        plt.plot(fringe[~good_ones, 0], fringe[~good_ones, 1], 'x',
                 color=colores[i % len(colores)])
        x_fit = slope * fringe[:, 1] + intercepts[i]
        plt.plot(x_fit, fringe[:, 1], color=colores[i % len(colores)], linestyle='--')
    plt.legend()
    if SAVE_RESULTS:
        output_dir = os.path.join(os.path.dirname(image_path), RESULTS_DIR)
        os.makedirs(output_dir, exist_ok=True)
        file_name = os.path.splitext(os.path.basename(image_path))[0]
        plt.savefig(os.path.join(output_dir, f"{file_name}.svg"))
    if SHOW_EACH_RESULT:
        plt.show()
    plt.close()

    return interfringe_distance, flecha


def plot_flechas_interfranjas(flechas, interfranjas, error_flechas=2.0, save_path=None):
    pendiente, intercepto, r_value, p_value, std_err = linregress(interfranjas, flechas)
    x = np.linspace(interfranjas.min(), interfranjas.max(), 4)
    y = pendiente * x + intercepto
    pendiente_u = ufloat(pendiente, std_err)
    plt.figure(figsize=(8, 6))
    plt.title(f"Flechas vs Interfranjas. Pendiente: {pendiente_u} (k=1)")
    plt.errorbar(interfranjas, flechas, yerr=error_flechas, fmt='s', color='black',
                 capsize=4, elinewidth=1.2, ecolor='gray')
    plt.plot(x, y, linestyle='--', color='black', linewidth=1.5)
    plt.xlabel("Interfranjas [px]")
    plt.ylabel("Flechas [px]")
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    # Ruta de tu imagen o directorio con imágenes que comienzan con un número
    image_path = r"/home/pablo/OneDrive/Documentos/INTI-Calibraciones/Planos/INTI Rosario 2024/Plano/Camara alejada/Transmision_SO"  # noqa: E501

    # Ejecutar el análisis
    analyze_dir_or_image(image_path)
