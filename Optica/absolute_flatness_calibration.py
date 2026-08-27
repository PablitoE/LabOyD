import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_erosion, rotate
from scipy.optimize import least_squares
from scipy.signal import find_peaks
from zernike import RZern

from FlechaInterfranja.interferogram_generation import FlatInterferogramGenerator

IMAGE_SHAPE = (256, 256)
WAVELENGTH = 632.8  # Wavelength in nm
ROTATION_FOURTH_IMAGE_DEG = 60.0  # Rotation of the fourth image in degrees

PITCH = 65e-6  # Pitch in meters
N_FRINGES = 5  # Number of fringes
DIAMETER = 15e-3  # Diameter in meters
MAX_FRINGES_ROTATION = 30.0  # Maximum rotation in degrees of the fringes
VISIBILITY_RATIO = 1.0  # Visibility ratio
MAX_DEVIATION_NM = 100.0  # Maximum deviation in nm

ORDER_PHASE = 3  # Maximum order of Zernike polynomials for phase
ORDER_VISIBILITY = 0  # Maximum order of Zernike polynomials for visibility
ORDER_BRIGHTNESS = 0  # Maximum order of Zernike polynomials for brightness
TOLERANCE_ESTIMATED_TILT_PERCENTAGE = 25
TOLERANCE_TILT_PERCENTAGE = 10 # Tolerance for tilt coefficients in the least squares fitting (in percentage)
MIN_TOLERANCE_TILT_RAD = 0.5

RANDOM_SEED = 0

PLOT_INTERFEROGRAMS = False
PLOT_MEASURED_SURFACES = False
PLOT_RESULTING_SURFACES = True
DEBUG_FRITZ = True
MAKE_SURFACES_AS_ZERNIKES = True


def estimate_frequency_from_array(arr, prominence=0.1, center=None, diameter=None, debug=False):
    """
    Use find_peaks to estimate the frequency of a 1D array. Use the prominence of the peaks to filter out noise.
    Get the threshold of the prominence as a fraction of the peak to peak amplitude.
    The frequency is estimated as the inverse of the average distance between peaks.
    """
    peaks_max, _ = find_peaks(arr, prominence=prominence, width=2, distance=5)
    peaks_min, _ = find_peaks(-arr, prominence=prominence, width=2, distance=5)
    if diameter is not None:
        center = arr.size // 2 if center is None else center
        peaks_max = peaks_max[np.abs(peaks_max - center) < diameter / 2 * 0.95]
        peaks_min = peaks_min[np.abs(peaks_min - center) < diameter / 2 * 0.95]
        max_width = diameter
    else:
        all_peaks = np.concatenate((peaks_max, peaks_min))
        max_width = np.ptp(all_peaks)
        if peaks_max[0] > peaks_min[0]:
            peaks_min = peaks_min[1:]
        else:
            peaks_max = peaks_max[1:]
        if peaks_max[-1] < peaks_min[-1]:
            peaks_min = peaks_min[:-1]
        else:
            peaks_max = peaks_max[:-1]
    peaks_max = clean_repeated_peaks(peaks_max, arr.size)
    peaks_min = clean_repeated_peaks(peaks_min, arr.size)
    if debug:
        plt.plot(arr)
        plt.plot(peaks_max, arr[peaks_max], 'ro')
        plt.plot(peaks_min, arr[peaks_min], 'go')
        plt.show()
    sorted_peaks = np.sort(np.concatenate((peaks_max, peaks_min)))
    if len(sorted_peaks) > 2:
        estimated_frequency = 1.0 / (2 * np.mean(np.diff(sorted_peaks)))
    else:
        estimated_frequency = 1 / max_width
    return estimated_frequency


def clean_repeated_peaks(peaks, length_array):
    if len(peaks) <= 1:
        return peaks
    elif len(peaks) == 2:
        if np.abs(peaks[0] - peaks[1]) < length_array / 2 * 0.8:
            return np.array([np.mean(peaks)]).astype(int)
        else:
            return peaks
    diff_peaks = np.diff(peaks)
    mean_diff = np.mean(diff_peaks)
    len_peaks = len(peaks)
    delete_peaks = []
    for i in range(1, len_peaks):
        if diff_peaks[i - 1] < mean_diff * 0.5:
            peaks[i - 1] = int((peaks[i - 1] + peaks[i]) / 2)
            delete_peaks.append(i)
    peaks = np.delete(peaks, delete_peaks)
    return peaks

def estimate_direction_by_gradients(interferogram, diameter):
    grad = list(np.gradient(interferogram))
    mask_valid = np.logical_not(np.logical_or(np.isnan(grad[0]), np.isnan(grad[1])))
    x, y = np.meshgrid(np.arange(interferogram.shape[1]), np.arange(interferogram.shape[0]))
    inside_diameter = np.sqrt((x - interferogram.shape[1] // 2) ** 2 + (y - interferogram.shape[0] // 2) ** 2) < (
        diameter / 2 - 3
    )
    mask_valid = np.logical_and(mask_valid, inside_diameter)
    grad[0] = grad[0][mask_valid]
    grad[1] = grad[1][mask_valid]
    # Bring the gradients to quadrants I and IV. Gradients in quadrants II and III are reversed.
    grad[1][grad[0] < 0] *= -1
    grad[0] = np.abs(grad[0])
    mean_grad = np.array([np.mean(grad[0]), np.mean(grad[1])])
    return mean_grad / np.linalg.norm(mean_grad)


def eval_interferogram_model(zernike_coeffs, zernike_surface: RZern, zernike_visibility: RZern,
                             zernike_brightness: RZern):
    zernike_surface_coeffs = zernike_coeffs[:zernike_surface.nk]
    zernike_visibility_coeffs = zernike_coeffs[zernike_surface.nk:zernike_surface.nk + zernike_visibility.nk]
    zernike_brightness_coeffs = zernike_coeffs[zernike_surface.nk + zernike_visibility.nk:]

    fitted_surface = zernike_surface.eval_grid(zernike_surface_coeffs, matrix=True)
    fitted_visibility = zernike_visibility.eval_grid(zernike_visibility_coeffs, matrix=True)
    fitted_brightness = zernike_brightness.eval_grid(zernike_brightness_coeffs, matrix=True)
    fitted_interferogram = fitted_brightness * (1 + fitted_visibility * np.cos(fitted_surface))
    return fitted_interferogram


def get_residuals(zernike_coeffs, zernike_surface: RZern, zernike_visibility: RZern, zernike_brightness: RZern,
                  interferogram: np.ndarray, plot=False):
    fitted_interferogram = eval_interferogram_model(zernike_coeffs,
                                                    zernike_surface, zernike_visibility, zernike_brightness)
    mask = np.isnan(fitted_interferogram)

    if plot:
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(interferogram)
        axs[0].set_title('Interferogram')
        axs[1].imshow(fitted_interferogram)
        axs[1].set_title('Fitted interferogram')
        plt.show()

    # Calculate residuals
    residuals = interferogram[~mask] - fitted_interferogram[~mask]
    return residuals.flatten()

def zernike_fit_interferogram(interferogram, max_order_phase, max_order_visibility, max_order_brightness, diameter,
                              plot=False):
    max_x = interferogram.shape[1] / diameter
    arr_x = np.linspace(-max_x, max_x, interferogram.shape[1])
    max_y = interferogram.shape[0] / diameter
    arr_y = np.linspace(-max_y, max_y, interferogram.shape[0])
    mat_x, mat_y = np.meshgrid(arr_x, arr_y)

    # Get a first estimate of the fringe frequency from the interferogram
    peak_prominence = 0.1 * np.ptp(interferogram)
    frequency_estimate_x = estimate_frequency_from_array(interferogram[interferogram.shape[0] // 2, :], peak_prominence,
                                                         diameter=diameter, debug=False)
    frequency_estimate_y = estimate_frequency_from_array(interferogram[:, interferogram.shape[1] // 2], peak_prominence,
                                                         diameter=diameter, debug=False)
    print(f"Frequency estimate: {frequency_estimate_x:.3f}, {frequency_estimate_y:.3f}")
    n_fringes_x = frequency_estimate_x * diameter
    n_fringes_y = frequency_estimate_y * diameter
    coeffs_tilt_estimate = np.array([n_fringes_x, n_fringes_y]) / 4

    # Update the signs of the frequencies by sampling the gradient of the interferogram in several places
    estimated_direction_of_fringes = estimate_direction_by_gradients(interferogram, diameter)
    coeffs_tilt_estimate = coeffs_tilt_estimate * np.sign(estimated_direction_of_fringes)

    # Start fit with order 1
    z_tilt = RZern(1)
    z_tilt.make_cart_grid(mat_x, mat_y)
    zernike_coeffs_tilt = np.zeros(z_tilt.nk)
    zernike_coeffs_tilt[1:] = 2 * np.pi * coeffs_tilt_estimate  # Initialize tilt coefficients
    z_visibility = RZern(max_order_visibility)
    z_visibility.make_cart_grid(mat_x, mat_y)
    zernike_coeffs_visibility = np.zeros(z_visibility.nk)
    zernike_coeffs_visibility[0] = np.ptp(interferogram) / 2  # Initialize visibility coefficient
    z_brightness = RZern(max_order_brightness)
    z_brightness.make_cart_grid(mat_x, mat_y)
    zernike_coeffs_brightness = np.zeros(z_brightness.nk)
    zernike_coeffs_brightness[0] = np.mean(interferogram)  # Initialize brightness coefficient

    coeffs0 = np.r_[zernike_coeffs_tilt, zernike_coeffs_visibility, zernike_coeffs_brightness]
    bounds = (-np.inf * np.ones_like(coeffs0), np.inf * np.ones_like(coeffs0))
    tolerance_tilt_x = abs(zernike_coeffs_tilt[1]) * TOLERANCE_TILT_PERCENTAGE / 100
    tolerance_tilt_x = max(tolerance_tilt_x, MIN_TOLERANCE_TILT_RAD)
    tolerance_tilt_y = abs(zernike_coeffs_tilt[2]) * TOLERANCE_TILT_PERCENTAGE / 100
    tolerance_tilt_y = max(tolerance_tilt_y, MIN_TOLERANCE_TILT_RAD)
    bounds[0][1] = max(0, coeffs0[1] - tolerance_tilt_x)
    bounds[1][1] = coeffs0[1] + tolerance_tilt_x
    bounds[0][2] = coeffs0[2] - tolerance_tilt_y
    bounds[1][2] = coeffs0[2] + tolerance_tilt_y

    initial_residual = get_residuals(coeffs0, z_tilt, z_visibility, z_brightness, interferogram, False)
    print(f"Initial MSE with rough estimate of tilt: {np.mean(initial_residual**2)}")

    least_squares_result = least_squares(get_residuals, coeffs0,
                                         args=(z_tilt, z_visibility, z_brightness, interferogram, False), bounds=bounds)
    print(f"Tilt estimated. MSE: {np.mean(least_squares_result.fun**2)}")
    print(f"Estimated frequencies: {np.array(least_squares_result.x[1:z_tilt.nk]) * 2 / np.pi / diameter}")

    fitted_interferogram = eval_interferogram_model(least_squares_result.x, z_tilt, z_visibility, z_brightness)
    if plot:
        fig, axs = plt.subplots(1, 4, figsize=(15, 5))
        axs[0].imshow(interferogram, cmap="gray", vmin=0, vmax=255)
        axs[0].set_title('Interferogram')
        axs[1].imshow(fitted_interferogram, cmap="gray")
        axs[1].set_title('Fitted interferogram with flat fringes')
        plt.show(block=False)

    # Repeat with estimated tilt coefficients
    zernike_coeffs_tilt = least_squares_result.x[:z_tilt.nk]
    zernike_coeffs_visibility = least_squares_result.x[z_tilt.nk:z_tilt.nk + z_visibility.nk]
    zernike_coeffs_brightness = least_squares_result.x[z_tilt.nk + z_visibility.nk:]

    z_surface = RZern(max_order_phase)
    z_surface.make_cart_grid(mat_x, mat_y)
    zernike_coeffs_surface = np.zeros(z_surface.nk)
    zernike_coeffs_surface[:z_tilt.nk] = zernike_coeffs_tilt
    coeffs0 = np.r_[zernike_coeffs_surface, zernike_coeffs_visibility, zernike_coeffs_brightness]

    bounds = (-np.inf * np.ones_like(coeffs0), np.inf * np.ones_like(coeffs0))
    # Do not allow the tilt coefficients to get too different from the initial estimate
    bounds[0][1] = coeffs0[1] - abs(coeffs0[1]) * TOLERANCE_TILT_PERCENTAGE / 100
    bounds[1][1] = coeffs0[1] + abs(coeffs0[1]) * TOLERANCE_TILT_PERCENTAGE / 100
    bounds[0][2] = coeffs0[2] - abs(coeffs0[2]) * TOLERANCE_TILT_PERCENTAGE / 100
    bounds[1][2] = coeffs0[2] + abs(coeffs0[2]) * TOLERANCE_TILT_PERCENTAGE / 100

    least_squares_result = least_squares(get_residuals, coeffs0,
                                         args=(z_surface, z_visibility, z_brightness, interferogram, False),
                                         bounds=bounds)

    print(f"Final MSE: {np.mean(least_squares_result.fun**2)}")

    zernike_coeffs_surface = least_squares_result.x[:z_surface.nk].copy()
    zernike_coeffs_visibility = least_squares_result.x[z_surface.nk:z_surface.nk + z_visibility.nk]
    zernike_coeffs_brightness = least_squares_result.x[z_surface.nk + z_visibility.nk:]

    zernike_coeffs_surface[:3] = 0.0  # Set piston and tilt coefficients to zero
    fitted_surface = z_surface.eval_grid(zernike_coeffs_surface, matrix=True)
    if plot:
        axs[2].imshow(fitted_surface, cmap="gray")
        axs[2].set_title('Fitted Surface')

        fitted_interferogram = eval_interferogram_model(least_squares_result.x, z_surface, z_visibility, z_brightness)
        axs[3].imshow(fitted_interferogram, cmap="gray")
        axs[3].set_title('Fitted Interferogram')
        plt.show()

    return zernike_coeffs_surface, zernike_coeffs_visibility, zernike_coeffs_brightness, fitted_surface


def zern_from_nk(nk):
    order = int((np.sqrt(8 * nk + 1) - 3) / 2)  # Calculate the order from the number of coefficients
    return RZern(order)


def zernike_surface(zernike_coeffs, shape, diameter):
    max_x = shape[1] / diameter
    arr_x = np.linspace(-max_x, max_x, shape[1])
    max_y = shape[0] / diameter
    arr_y = np.linspace(-max_y, max_y, shape[0])
    mat_x, mat_y = np.meshgrid(arr_x, arr_y)

    z_surface = zern_from_nk(len(zernike_coeffs))
    z_surface.make_cart_grid(mat_x, mat_y)
    fitted_surface = z_surface.eval_grid(zernike_coeffs, matrix=True)
    return fitted_surface


def zernike_fit_lsq(order, surface, diameter):
    max_x = surface.shape[1] / diameter
    arr_x = np.linspace(-max_x, max_x, surface.shape[1])
    max_y = surface.shape[0] / diameter
    arr_y = np.linspace(-max_y, max_y, surface.shape[0])
    mat_x, mat_y = np.meshgrid(arr_x, arr_y)

    z_surface = RZern(order)
    z_surface.make_cart_grid(mat_x, mat_y)
    return z_surface.fit_cart_grid(surface)[0]


def fit_interferogram_with_zernikes(interferogram, max_order_phase: int=4, max_order_visibility: int=4,
                                    max_order_brightness: int=4, diameter_px: float=None):
    """
    Fit the interferogram with Zernike polynomials up to a specified order.

    Parameters:
    - interferogram: 2D numpy array representing the interferogram.
    - max_order_phase: Maximum order of Zernike polynomials for the phase component.
    - max_order_visibility: Maximum order of Zernike polynomials for the visibility component.
    - max_order_brightness: Maximum order of Zernike polynomials for the brightness component.
    - diameter_px: Diameter of the circular aperture in pixels. If None, it will be estimated from the interferogram.

    Returns:
    - zernike_coeffs: Coefficients of the fitted Zernike polynomials.
    - fitted_surface: The surface reconstructed from the fitted Zernike coefficients.
    """
    if diameter_px is None:
        # Estimate diameter from the interferogram
        threshold = np.mean(interferogram) + 0.5 * np.std(interferogram)
        mask = interferogram > threshold
        y_indices, x_indices = np.where(mask)
        diameter_px = max(x_indices.max() - x_indices.min(), y_indices.max() - y_indices.min())

    # Fit Zernike polynomials
    zernike_coeffs_surface, _, _, fitted_surface = zernike_fit_interferogram(
        interferogram, max_order_phase, max_order_visibility, max_order_brightness, diameter=diameter_px
    )

    # Reconstruct surface from Zernike coefficients
    # fitted_surface = zernike_surface(zernike_coeffs_surface, IMAGE_SHAPE, diameter=diameter_px)
    fitted_surface = fitted_surface / (2 * np.pi)

    return zernike_coeffs_surface, fitted_surface


def filter_as_zernike(surface, order, diameter_px):
    zern_coeffs = zernike_fit_lsq(order, surface, diameter_px)
    surface = zernike_surface(zern_coeffs, surface.shape, diameter_px)
    surface[np.isnan(surface)] = 0
    return surface, zern_coeffs


def fritz_algorithm(z_d, z_e, z_f, z_g, rotation_rad):
    max_coeff = len(z_d)
    zern = zern_from_nk(max_coeff)
    z_m = np.zeros(max_coeff)
    z_k = np.zeros(max_coeff)
    z_l = np.zeros(max_coeff)
    z_m_done = np.zeros(max_coeff, dtype=bool)
    noll_k = 0
    working_in_cycle = True
    while not np.all(z_m_done):
        if noll_k == 0:
            if not working_in_cycle:
                raise ValueError("Algorithm failed")
            working_in_cycle = False

        if z_m_done[noll_k]:
            noll_k = noll_k + 1 if noll_k < max_coeff - 1 else 0
            continue
        working_in_cycle = True

        zd = z_d[noll_k]
        ze = z_e[noll_k]
        zf = z_f[noll_k]
        zg = z_g[noll_k]

        n, m = zern.noll2nm(noll_k + 1)
        c_n = np.cos(m * rotation_rad)
        s_n = np.sin(abs(m) * rotation_rad)     # Not sure if this is correct
        noll_neg_m = zern.nm2noll(n, -m) - 1    # -1 is to convert Noll index to array index
        if n % 2 == 1:
            if m < 0:
                class_fritz = 2
            elif m > 0:
                class_fritz = 3
            else:
                raise ValueError("Invalid m value")
        else:
            if m == 0:
                class_fritz = 1
            elif m > 0:
                class_fritz = 2
            else:
                class_fritz = 4
        if abs(s_n) < 1e-9:
            # Special cases sin(m*rotation_rad) ~= 0
            if (class_fritz == 3 and c_n > 0) or class_fritz == 4:
                class_fritz = 5
            elif class_fritz == 3 and c_n < 0:
                class_fritz = 6

        if class_fritz == 1:
            z_m[noll_k] = (ze + zf) / 4 + (zg - zd) / 2
            z_m_done[noll_k] = True
            z_k[noll_k] = zg - z_m[noll_k]
            z_l[noll_k] = (ze + zf) / 2 - z_m[noll_k]
        elif class_fritz == 2:
            z_m[noll_k] = (-zd + ze + zg) / 2
            z_m_done[noll_k] = True
            z_k[noll_k] = zg - z_m[noll_k]
            z_l[noll_k] = ze - z_m[noll_k]
        elif class_fritz == 3:
            if z_m_done[noll_neg_m]:
                z_m[noll_k] = (z_f[noll_neg_m] - z_l[noll_neg_m] - z_m[noll_neg_m] * c_n) / s_n
                z_m_done[noll_k] = True
                z_k[noll_k] = z_m[noll_k] - zg
                z_l[noll_k] = z_m[noll_k] - (ze + zd + zg) / 2
        elif class_fritz == 4:
            if z_m_done[noll_neg_m]:
                z_m[noll_k] = (-z_f[noll_neg_m] + z_l[noll_neg_m] + z_m[noll_neg_m] * c_n) / s_n
                z_m_done[noll_k] = True
                z_k[noll_k] = z_m[noll_k] - zg
                z_l[noll_k] = z_m[noll_k] - (ze + zd + zg) / 2
        elif class_fritz == 6:
            pseudo_inverse = np.array([[3, 0, -3, -3],[-1, -2, -3, -1],[1, 2, -3, 1]]) / 6
            zs = pseudo_inverse @ np.array([zd, ze, zf, zg])
            z_k[noll_k] = zs[0]
            z_l[noll_k] = zs[1]
            z_m[noll_k] = zs[2]
            z_m_done[noll_k] = True
        else:
            matrix_a = np.array([
                [ 1, -1,  0],
                [ 0, -1,  1],
                [ 0, -1,  1],
                [-1,  0,  1]
            ])
            b = np.array([zd, ze, zf, zg])
            zs = np.linalg.lstsq(matrix_a, b, rcond=None)[0]
            z_k[noll_k] = zs[0]
            z_l[noll_k] = zs[1]
            z_m[noll_k] = zs[2]
            z_m_done[noll_k] = True

        noll_k = noll_k + 1 if noll_k < max_coeff - 1 else 0
    return z_m, z_k, z_l


def mirror_x_zernike_coeffs(zernike_coeffs):
    zern = zern_from_nk(len(zernike_coeffs))
    mirrored_zernike_coeffs = np.copy(zernike_coeffs)
    for i in range(len(zernike_coeffs)):
        n, m = zern.noll2nm(i + 1)
        if (n % 2 == 1 and m > 0) or (n % 2 == 0 and m < 0):
            mirrored_zernike_coeffs[i] = -mirrored_zernike_coeffs[i]
    return mirrored_zernike_coeffs


def rotate_zernike_coeffs(zernike_coeffs, angle_rad):
    zern = zern_from_nk(len(zernike_coeffs))
    rotated_zernike_coeffs = np.copy(zernike_coeffs)
    for i in range(len(zernike_coeffs)):
        n, m = zern.noll2nm(i + 1)
        neg_m_index = zern.nm2noll(n, -m) - 1
        if m != 0:
            rotated_zernike_coeffs[i] = zernike_coeffs[i] * np.cos(m * angle_rad) + (
                zernike_coeffs[neg_m_index] * np.sin(m * angle_rad))
    return rotated_zernike_coeffs


if __name__ == "__main__":
    generator = FlatInterferogramGenerator(shape=IMAGE_SHAPE, wavelength_nm=WAVELENGTH, pixel_size=PITCH,
                                           min_fringe=N_FRINGES, max_fringe=N_FRINGES, diameter=DIAMETER,
                                           max_rotation=MAX_FRINGES_ROTATION, visibility_ratio=VISIBILITY_RATIO,
                                           seed=RANDOM_SEED)
    frequency = N_FRINGES / generator.diameter_pixels

    max_deviationA = np.random.rand() * MAX_DEVIATION_NM
    max_deviationB = np.random.rand() * MAX_DEVIATION_NM
    max_deviationC = np.random.rand() * MAX_DEVIATION_NM
    generator.current_maximum_deviation_nm = max_deviationA
    surfaceA = generator.simulate_surface()      # Surface A (k in Fritz algorithm)
    generator.current_maximum_deviation_nm = max_deviationB
    surfaceB = generator.simulate_surface()      # Surface B (l in Fritz algorithm)
    generator.current_maximum_deviation_nm = max_deviationC
    surfaceC = generator.simulate_surface()      # Surface C (m in Fritz algorithm)

    surfaceA_filtered, zern_coeffs_A = filter_as_zernike(surfaceA, ORDER_PHASE, generator.diameter_pixels)
    surfaceB_filtered, zern_coeffs_B = filter_as_zernike(surfaceB, ORDER_PHASE, generator.diameter_pixels)
    surfaceC_filtered, zern_coeffs_C = filter_as_zernike(surfaceC, ORDER_PHASE, generator.diameter_pixels)
    if MAKE_SURFACES_AS_ZERNIKES:
        surfaceA = surfaceA_filtered
        surfaceB = surfaceB_filtered
        surfaceC = surfaceC_filtered

    # Process a single image per pair of surfaces
    surface_BA = surfaceA + np.fliplr(surfaceB)  # Measurement D in Fritz algorithm
    generator.surface = surface_BA
    interferogram_BA = generator.generate_flat_interferogram(normalized_carrier_frequency=frequency)
    frequencies_BA = generator.current_rotated_frequencies

    surface_BC = surfaceC + np.fliplr(surfaceB)  # Measurement E in Fritz algorithm
    generator.surface = surface_BC
    interferogram_BC = generator.generate_flat_interferogram(normalized_carrier_frequency=frequency)
    frequencies_BC = generator.current_rotated_frequencies

    surface_AC = surfaceC + np.fliplr(surfaceA)  # Measurement G in Fritz algorithm
    generator.surface = surface_AC
    interferogram_AC = generator.generate_flat_interferogram(normalized_carrier_frequency=frequency)
    frequencies_AC = generator.current_rotated_frequencies

    surface_BCrot = rotate(surfaceC, ROTATION_FOURTH_IMAGE_DEG, reshape=False) + np.fliplr(surfaceB)
    generator.surface = surface_BCrot            # Measurement F in Fritz algorithm
    interferogram_BCrot = generator.generate_flat_interferogram(normalized_carrier_frequency=frequency)
    frequencies_BCrot = generator.current_rotated_frequencies

    if PLOT_INTERFEROGRAMS:
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        axs[0, 0].imshow(interferogram_BA, cmap='gray')
        axs[0, 0].set_title('Interferogram A-B')
        axs[0, 1].imshow(interferogram_BC, cmap='gray')
        axs[0, 1].set_title('Interferogram B-C')
        axs[1, 0].imshow(interferogram_AC, cmap='gray')
        axs[1, 0].set_title('Interferogram A-C')
        axs[1, 1].imshow(interferogram_BCrot, cmap='gray')
        axs[1, 1].set_title('Interferogram B-C rotated')
        plt.show()

    rotation_fourth_image_rad = np.deg2rad(ROTATION_FOURTH_IMAGE_DEG)
    if DEBUG_FRITZ:
        zernike_coeffs_BA = zernike_fit_lsq(ORDER_PHASE, surface_BA, generator.diameter_pixels)
        zernike_coeffs_BC = zernike_fit_lsq(ORDER_PHASE, surface_BC, generator.diameter_pixels)
        zernike_coeffs_AC = zernike_fit_lsq(ORDER_PHASE, surface_AC, generator.diameter_pixels)
        zernike_coeffs_BCrot = zernike_fit_lsq(ORDER_PHASE, surface_BCrot, generator.diameter_pixels)

        mirrored_zern_coeffs_B = mirror_x_zernike_coeffs(zern_coeffs_B)
        expected_zern_BA = zern_coeffs_A + mirrored_zern_coeffs_B
        expected_zern_BC = zern_coeffs_C + mirrored_zern_coeffs_B
        expected_zern_AC = zern_coeffs_C + mirror_x_zernike_coeffs(zern_coeffs_A)
        rotated_zern_coeffs_C = rotate_zernike_coeffs(zern_coeffs_C, rotation_fourth_image_rad)
        expected_zern_BCrot = rotated_zern_coeffs_C + mirrored_zern_coeffs_B
    else:
        print(f"{'-'*50}\nFrequencies A-B: {frequencies_BA}")
        zernike_coeffs_BA, fitted_surface_BA = fit_interferogram_with_zernikes(
            interferogram_BA, ORDER_PHASE, ORDER_VISIBILITY, ORDER_BRIGHTNESS, diameter_px=generator.diameter_pixels
        )
        print(f"{'-'*50}\nFrequencies B-C: {frequencies_BC}")
        zernike_coeffs_BC, fitted_surface_BC = fit_interferogram_with_zernikes(
            interferogram_BC, ORDER_PHASE, ORDER_VISIBILITY, ORDER_BRIGHTNESS, diameter_px=generator.diameter_pixels
        )
        print(f"{'-'*50}\nFrequencies A-C: {frequencies_AC}")
        zernike_coeffs_AC, fitted_surface_AC = fit_interferogram_with_zernikes(
            interferogram_AC, ORDER_PHASE, ORDER_VISIBILITY, ORDER_BRIGHTNESS, diameter_px=generator.diameter_pixels
        )
        print(f"{'-'*50}\nFrequencies B-C rotated: {frequencies_BCrot}")
        zernike_coeffs_BCrot, fitted_surface_BCrot = fit_interferogram_with_zernikes(
            interferogram_BCrot, ORDER_PHASE, ORDER_VISIBILITY, ORDER_BRIGHTNESS, diameter_px=generator.diameter_pixels
        )

        if PLOT_MEASURED_SURFACES:
            fig, axs = plt.subplots(2, 4, figsize=(10, 10))
            axs[0, 0].imshow(fitted_surface_BA, cmap='gray')
            axs[0, 0].set_title('Fitted surface A-B')
            axs[0, 1].imshow(fitted_surface_BC, cmap='gray')
            axs[0, 1].set_title('Fitted surface B-C')
            axs[0, 2].imshow(fitted_surface_AC, cmap='gray')
            axs[0, 2].set_title('Fitted surface A-C')
            axs[0, 3].imshow(fitted_surface_BCrot, cmap='gray')
            axs[0, 3].set_title('Fitted surface B-C rotated')
            axs[1, 0].imshow(surface_BA, cmap='gray')
            axs[1, 0].set_title('Surface A-B')
            axs[1, 1].imshow(surface_BC, cmap='gray')
            axs[1, 1].set_title('Surface B-C')
            axs[1, 2].imshow(surface_AC, cmap='gray')
            axs[1, 2].set_title('Surface A-C')
            axs[1, 3].imshow(surface_BCrot, cmap='gray')
            axs[1, 3].set_title('Surface B-C rotated')
            plt.show()

    z_C, z_A, z_B = fritz_algorithm(zernike_coeffs_BA, zernike_coeffs_BC, zernike_coeffs_BCrot, zernike_coeffs_AC,
                                    rotation_rad=-rotation_fourth_image_rad)

    # Evaluate the results
    print(f"Error in Surface A zernike coeffs: {abs(zern_coeffs_A - z_A)}")
    print(f"Error in Surface B zernike coeffs: {abs(zern_coeffs_B - z_B)}")
    print(f"Error in Surface C zernike coeffs: {abs(zern_coeffs_C - z_C)}")
    print(f"{'-'*50}")

    shrunk_mask = generator.aperture_mask.copy()
    shrunk_mask = binary_erosion(shrunk_mask, structure=np.ones((3, 3)), iterations=2)
    result_A = zernike_surface(z_A, IMAGE_SHAPE, generator.diameter_pixels)
    error_A_nm = (result_A - surfaceA) * WAVELENGTH / 2
    print(f"RMSE A: {np.sqrt(np.mean(error_A_nm[shrunk_mask]**2)):.2f} nm")
    result_B = zernike_surface(z_B, IMAGE_SHAPE, generator.diameter_pixels)
    error_B_nm = (result_B - surfaceB) * WAVELENGTH / 2
    print(f"RMSE B: {np.sqrt(np.mean(error_B_nm[shrunk_mask]**2)):.2f} nm")
    result_C = zernike_surface(z_C, IMAGE_SHAPE, generator.diameter_pixels)
    error_C_nm = (result_C - surfaceC) * WAVELENGTH / 2
    print(f"RMSE C: {np.sqrt(np.mean(error_C_nm[shrunk_mask]**2)):.2f} nm")

    if PLOT_RESULTING_SURFACES:
        fig, axs = plt.subplots(3, 3, figsize=(15, 5))
        axs[0, 0].imshow(result_A, cmap='gray')
        axs[0, 0].set_title(r'Resulting surface A / $\lambda / 2$')
        axs[0, 1].imshow(result_B, cmap='gray')
        axs[0, 1].set_title(r'Resulting surface B / $\lambda / 2$')
        axs[0, 2].imshow(result_C, cmap='gray')
        axs[0, 2].set_title(r'Resulting surface C / $\lambda / 2$')
        axs[1, 0].imshow(surfaceA, cmap='gray')
        axs[1, 0].set_title(r'Surface A / $\lambda / 2$')
        axs[1, 1].imshow(surfaceB, cmap='gray')
        axs[1, 1].set_title(r'Surface B / $\lambda / 2$')
        axs[1, 2].imshow(surfaceC, cmap='gray')
        axs[1, 2].set_title(r'Surface C / $\lambda / 2$')
        axs[2, 0].imshow(error_A_nm, cmap='gray')
        axs[2, 0].set_title('Error A / nm')
        axs[2, 1].imshow(error_B_nm, cmap='gray')
        axs[2, 1].set_title('Error B / nm')
        axs[2, 2].imshow(error_C_nm, cmap='gray')
        axs[2, 2].set_title('Error C / nm')
        plt.show()
