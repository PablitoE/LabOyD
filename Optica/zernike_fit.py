from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import OptimizeResult, least_squares
from scipy.signal import find_peaks, savgol_filter
from zernike import RZern

TOLERANCE_ESTIMATED_TILT_PERCENTAGE = 25
TOLERANCE_TILT_PERCENTAGE = 10 # Tolerance for tilt coefficients in the least squares fitting (in percentage)
MIN_TOLERANCE_TILT_RAD = 0.5

MAX_LS_ITERATIONS_PER_COEFF = 5
PROGRESSIVE_ORDER_INCREASE = False

DEBUG_FREQUENCY_ESTIMATE = False
PLOT_HISTORY_LEAST_SQUARES = False


def estimate_frequency_from_array(arr, prominence=0.1, center=None, diameter=None, debug=False, distance=5):
    """
    Use find_peaks to estimate the frequency of a 1D array. Use the prominence of the peaks to filter out noise.
    Get the threshold of the prominence as a fraction of the peak to peak amplitude.
    The frequency is estimated as the inverse of the average distance between peaks.
    """
    arr = savgol_filter(arr, window_length=distance*2, polyorder=3)
    peaks_max, _ = find_peaks(arr, prominence=prominence, width=2, distance=distance)
    peaks_min, _ = find_peaks(-arr, prominence=prominence, width=2, distance=distance)
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


def eval_interferogram_model_from_coeffs(surface_zernike_coeffs, visibility_zernike_coeffs, brightness_zernike_coeffs,
                                         shape, diameter):
    z_surface = zern_from_nk(len(surface_zernike_coeffs))
    z_visibility = zern_from_nk(len(visibility_zernike_coeffs))
    z_brightness = zern_from_nk(len(brightness_zernike_coeffs))
    mat_x, mat_y = mat_xy(shape, diameter)
    z_surface.make_cart_grid(mat_x, mat_y)
    z_visibility.make_cart_grid(mat_x, mat_y)
    z_brightness.make_cart_grid(mat_x, mat_y)
    return eval_interferogram_model(
        np.concatenate((surface_zernike_coeffs, visibility_zernike_coeffs, brightness_zernike_coeffs)),
        z_surface, z_visibility, z_brightness,
    )

def eval_interferogram_model(zernike_coeffs, zernike_surface: RZern, zernike_visibility: RZern,
                             zernike_brightness: RZern, get_jacobian=False):
    zernike_surface_coeffs = zernike_coeffs[:zernike_surface.nk]
    zernike_visibility_coeffs = zernike_coeffs[zernike_surface.nk:zernike_surface.nk + zernike_visibility.nk]
    zernike_brightness_coeffs = zernike_coeffs[zernike_surface.nk + zernike_visibility.nk:]

    fitted_surface = zernike_surface.eval_grid(zernike_surface_coeffs, matrix=True)
    fitted_visibility = zernike_visibility.eval_grid(zernike_visibility_coeffs, matrix=True)
    fitted_brightness = zernike_brightness.eval_grid(zernike_brightness_coeffs, matrix=True)
    if get_jacobian:
        mask = np.isnan(fitted_surface)
        fitted_surface = fitted_surface[~mask]
        fitted_visibility = fitted_visibility[~mask]
        fitted_brightness = fitted_brightness[~mask]
        # Get the Zernike polynomials evaluated on the grid for each component
        max_nk = max(zernike_surface.nk, zernike_visibility.nk, zernike_brightness.nk)
        zernike_pols = np.zeros((fitted_surface.size, max_nk))
        for k in range(max_nk):
            zernike_pols[:, k] = zernike_surface.eval_grid(np.eye(max_nk)[k], matrix=True)[~mask]
        # Compute the Jacobian matrix
        jac_surface = (
            (fitted_brightness * fitted_visibility * np.sin(fitted_surface))[:, np.newaxis]
            * zernike_pols[:, : zernike_surface.nk]
        )
        jac_visibility = -(
            (fitted_brightness * np.cos(fitted_surface))[:, np.newaxis] * zernike_pols[:, : zernike_visibility.nk]
        )
        jac_brightness = -(
            1 + fitted_visibility * np.cos(fitted_surface)
        )[:, np.newaxis] * zernike_pols[:, : zernike_brightness.nk]
        return np.concatenate((jac_surface, jac_visibility, jac_brightness), axis=1)
    else:
        fitted_interferogram = fitted_brightness * (1 + fitted_visibility * np.cos(fitted_surface))
        return fitted_interferogram


def zernike_radial_weights(zernike: RZern, p=1):
    """
    Compute the regularization weights for the Zernike coefficients based on their order.
    The weights are proportional to (n+1)^p, where n is the radial order of the Zernike polynomial.
    """
    weights = np.zeros(zernike.nk)
    for k in range(zernike.nk):
        n, m = zernike.noll2nm(k + 1)
        weights[k] = (m + 1) ** p
    return weights


def regularization_radial(zernike_surface: RZern, zernike_visibility: RZern, zernike_brightness: RZern, p=1,
                          n_images=1):
    weights_surface = zernike_radial_weights(zernike_surface, p)
    weights_visibility = zernike_radial_weights(zernike_visibility, p)
    weights_brightness = zernike_radial_weights(zernike_brightness, p)
    weights = np.abs(np.concatenate((np.repeat(weights_surface[:3], n_images - 1),
                                     weights_surface, np.repeat(weights_visibility, n_images),
                                     np.repeat(weights_brightness, n_images))))
    return weights / np.sum(weights)


def get_im_zernike_coeffs_indices(counter_image: int, zernike_surface: RZern, zernike_visibility: RZern,
                                  zernike_brightness: RZern, n_images=1):
    ind_visibility_start = zernike_surface.nk + (n_images - 1) * 3
    ind_brightness_start = ind_visibility_start + n_images * zernike_visibility.nk
    return np.concatenate((
            np.arange(counter_image * 3, counter_image * 3 + 3),  # Piston and tilt coefficients for this image
            np.arange(zernike_surface.nk - 3) + 3 * n_images,  # Surface coefficients (shared across images)
            np.arange(ind_visibility_start + counter_image * zernike_visibility.nk,
                      ind_visibility_start + (counter_image + 1) * zernike_visibility.nk),  # Visibility coefficients
            np.arange(ind_brightness_start + counter_image * zernike_brightness.nk,
                      ind_brightness_start + (counter_image + 1) * zernike_brightness.nk)  # Brightness coefficients
        ))


def get_residuals(zernike_coeffs, zernike_surface: RZern, zernike_visibility: RZern, zernike_brightness: RZern,
                  interferogram: np.ndarray, regularization_alpha=0, regularization_weights=None, n_images=1,
                  plot=False):
    if interferogram.ndim == 2:
        interferogram = np.expand_dims(interferogram, axis=0)
    residuals = np.array([])
    for counter_image in range(n_images):
        zernike_coeffs_indices = get_im_zernike_coeffs_indices(counter_image, zernike_surface, zernike_visibility,
                                                               zernike_brightness, n_images)
        fitted_interferogram = eval_interferogram_model(zernike_coeffs[zernike_coeffs_indices],
                                                        zernike_surface, zernike_visibility, zernike_brightness)
        mask = np.isnan(fitted_interferogram)

        if plot:
            fig, axs = plt.subplots(1, 2)
            axs[0].imshow(interferogram[counter_image])
            axs[0].set_title('Interferogram')
            axs[1].imshow(fitted_interferogram)
            axs[1].set_title('Fitted interferogram')
            plt.show()

        # Calculate residuals
        residuals = np.append(residuals, interferogram[counter_image][~mask] - fitted_interferogram[~mask])

    # Calculate regularization term
    if regularization_weights is not None:
        n_residuals = len(residuals)
        regularization_term = np.sqrt(
            np.sum((zernike_coeffs * regularization_weights) ** 2) * regularization_alpha * n_residuals
        )
        residuals = np.append(residuals, regularization_term)
    return residuals.flatten()

def get_jacobian(zernike_coeffs, zernike_surface: RZern, zernike_visibility: RZern, zernike_brightness: RZern,
                 interferogram=None, regularization_alpha=0, regularization_weights=None, n_images=1, plot=False):
    jacobian = np.array([])
    for counter_image in range(n_images):
        zernike_coeffs_indices = get_im_zernike_coeffs_indices(counter_image, zernike_surface, zernike_visibility,
                                                               zernike_brightness, n_images)
        j = eval_interferogram_model(zernike_coeffs[zernike_coeffs_indices], zernike_surface, zernike_visibility,
                                     zernike_brightness, get_jacobian=True)
        row_jacobian = np.zeros((j.shape[0], len(zernike_coeffs)))
        row_jacobian[:, zernike_coeffs_indices] = j
        jacobian = np.vstack((jacobian, row_jacobian)) if jacobian.size else row_jacobian
    if regularization_weights is not None:
        n_residuals = jacobian.shape[0] / n_images
        regularization_term = np.sqrt(
            np.sum((zernike_coeffs * regularization_weights) ** 2) * regularization_alpha * n_residuals
        )
        regularization_jacobian = (
            zernike_coeffs * regularization_weights**2 * regularization_alpha * n_residuals / regularization_term
        )
        jacobian = np.vstack((jacobian, regularization_jacobian))
    return jacobian

def callback_ls(intermediate_result: OptimizeResult|np.ndarray, history_costs: list, history_reg: list, args=None):
    if isinstance(intermediate_result, OptimizeResult):
        history_costs.append(intermediate_result.cost)
    elif isinstance(intermediate_result, np.ndarray):
        residuals = get_residuals(intermediate_result, *args)
        if history_reg is not None:
            history_reg.append(residuals[-1] **2)
            residuals = residuals[:-1]
        history_costs.append(np.sum(residuals ** 2))
    else:
        raise ValueError("intermediate_result must be either OptimizeResult or np.ndarray")


def print_if_verbose(verbose, *args):
    if verbose:
        print(*args)


def zernike_fit_interferogram(interferogram, max_order_phase, max_order_visibility, max_order_brightness, diameter,
                              progressive_order_increase=False, plot=False, verbose=False):
    if isinstance(interferogram, np.ndarray) and np.ndim(interferogram) == 2:
        interferogram = [interferogram]
    if isinstance(interferogram, list):
        interferogram = np.array(interferogram)
    mat_x, mat_y = mat_xy(interferogram[0].shape, diameter)
    n_images = interferogram.shape[0]

    # Get a first estimate of the fringe frequency from the interferograms
    z_tilt = RZern(1)
    z_tilt.make_cart_grid(mat_x, mat_y)
    zernike_coeffs_tilt = np.zeros((n_images, 3))
    z_brightness = RZern(max_order_brightness)
    z_brightness.make_cart_grid(mat_x, mat_y)
    zernike_coeffs_brightness = np.zeros((n_images, z_brightness.nk))
    z_visibility = RZern(max_order_visibility)
    z_visibility.make_cart_grid(mat_x, mat_y)
    zernike_coeffs_visibility = np.zeros((n_images, z_visibility.nk))
    peak_prominence = 0.1 * np.ptp(interferogram)
    for i, im in enumerate(interferogram):
        frequency_estimate_x = estimate_frequency_from_array(im[im.shape[0] // 2, :], peak_prominence,
                                                            diameter=diameter, debug=DEBUG_FREQUENCY_ESTIMATE)
        frequency_estimate_y = estimate_frequency_from_array(im[:, im.shape[1] // 2], peak_prominence,
                                                            diameter=diameter, debug=DEBUG_FREQUENCY_ESTIMATE)
        print_if_verbose(verbose, f"Im#{i}: Frequency estimate: {frequency_estimate_x:.3f}, {frequency_estimate_y:.3f}")
        n_fringes_x = frequency_estimate_x * diameter
        n_fringes_y = frequency_estimate_y * diameter
        coeffs_tilt_estimate = np.array([n_fringes_x, n_fringes_y]) / 4

        # Update the signs of the frequencies by sampling the gradient of the interferogram in several places
        estimated_direction_of_fringes = estimate_direction_by_gradients(im, diameter)
        coeffs_tilt_estimate = coeffs_tilt_estimate * np.sign(estimated_direction_of_fringes)

        # Start fit with order 1
        zernike_coeffs_tilt[i, 1:] = 2 * np.pi * coeffs_tilt_estimate  # Initialize tilt coefficients
        zernike_coeffs_brightness[i, 0] = np.mean(im)  # Initialize brightness coefficient
        # Initialize visibility coefficient
        zernike_coeffs_visibility[i, 0] = np.ptp(im) / 2 / zernike_coeffs_brightness[i, 0]

        coeffs0 = np.r_[zernike_coeffs_tilt[i], zernike_coeffs_visibility[i], zernike_coeffs_brightness[i]]
        bounds = (-np.inf * np.ones_like(coeffs0), np.inf * np.ones_like(coeffs0))
        tolerance_tilt_x = abs(zernike_coeffs_tilt[i, 1]) * TOLERANCE_TILT_PERCENTAGE / 100
        tolerance_tilt_x = max(tolerance_tilt_x, MIN_TOLERANCE_TILT_RAD)
        tolerance_tilt_y = abs(zernike_coeffs_tilt[i, 2]) * TOLERANCE_TILT_PERCENTAGE / 100
        tolerance_tilt_y = max(tolerance_tilt_y, MIN_TOLERANCE_TILT_RAD)
        bounds[0][1] = max(0, coeffs0[1] - tolerance_tilt_x)
        bounds[1][1] = coeffs0[1] + tolerance_tilt_x
        bounds[0][2] = coeffs0[2] - tolerance_tilt_y
        bounds[1][2] = coeffs0[2] + tolerance_tilt_y

        initial_residual = get_residuals(coeffs0, z_tilt, z_visibility, z_brightness, im, plot=False)
        print_if_verbose(verbose, f"Im#{i}: Initial RMSE with rough estimate of tilt: "
                                  f"{np.sqrt(np.mean(initial_residual**2))}")

        least_squares_result = least_squares(get_residuals, coeffs0,
                                            args=(z_tilt, z_visibility, z_brightness, im), bounds=bounds)
        print_if_verbose(verbose, f"Im#{i}: Tilt estimated. RMSE: {np.sqrt(np.mean(least_squares_result.fun**2))}")
        print_if_verbose(
            verbose, f'Estimated frequencies: {np.array(least_squares_result.x[1 : z_tilt.nk]) * 2 / np.pi / diameter}'
        )

        if plot:
            fitted_interferogram = eval_interferogram_model(least_squares_result.x, z_tilt, z_visibility, z_brightness)

            fig, axs = plt.subplots(1, 4, figsize=(15, 5))
            axs[0].imshow(im, cmap="gray", vmin=0, vmax=255)
            axs[0].set_title('Interferogram')
            axs[1].imshow(fitted_interferogram, cmap="gray")
            axs[1].set_title('Fitted interferogram with flat fringes')
            plt.show(block=False)

        zernike_coeffs_tilt[i] = least_squares_result.x[:z_tilt.nk]
        zernike_coeffs_visibility[i] = least_squares_result.x[z_tilt.nk:z_tilt.nk + z_visibility.nk]
        zernike_coeffs_brightness[i] = least_squares_result.x[z_tilt.nk + z_visibility.nk:]

    # Repeat with estimated tilt coefficients
    zernike_coeffs_previous = zernike_coeffs_tilt.flatten().copy()
    zernike_coeffs_visibility = zernike_coeffs_visibility.flatten().copy()
    zernike_coeffs_brightness = zernike_coeffs_brightness.flatten().copy()
    previous_nk = zernike_coeffs_previous.size

    start_order = 2 if progressive_order_increase else max_order_phase
    for progressive_order in range(start_order, max_order_phase + 1):
        z_surface = RZern(progressive_order)
        z_surface.make_cart_grid(mat_x, mat_y)
        zernike_coeffs_surface = np.zeros(z_surface.nk + (n_images - 1) * 3)
        zernike_coeffs_surface[:previous_nk] = zernike_coeffs_previous.copy()
        coeffs0 = np.r_[zernike_coeffs_surface, zernike_coeffs_visibility, zernike_coeffs_brightness]
        if REGULARIZATION_ALPHA > 0:
            regularization_weights = regularization_radial(z_surface, z_visibility, z_brightness, p=1,
                                                           n_images=n_images)
        else:
            regularization_weights = None

        bounds = (-np.inf * np.ones_like(coeffs0), np.inf * np.ones_like(coeffs0))
        # Do not allow the piston and tilt coefficients to get too different from the initial estimate
        bounds[0][: 3 * n_images] = (
            coeffs0[: 3 * n_images] - abs(coeffs0[: 3 * n_images]) * TOLERANCE_TILT_PERCENTAGE / 100
        )
        bounds[1][: 3 * n_images] = (
            coeffs0[: 3 * n_images] + abs(coeffs0[: 3 * n_images]) * TOLERANCE_TILT_PERCENTAGE / 100
        )

        history_costs = []
        history_reg = None if REGULARIZATION_ALPHA == 0 else []
        args = (z_surface, z_visibility, z_brightness, interferogram, REGULARIZATION_ALPHA, regularization_weights,
                n_images)
        if PLOT_HISTORY_LEAST_SQUARES:
            callback_in_for = partial(callback_ls, history_costs=history_costs, history_reg=history_reg, args=args)
        else:
            callback_in_for = None

        max_iterations = MAX_LS_ITERATIONS_PER_COEFF * len(coeffs0)
        optimization_result = least_squares(get_residuals, coeffs0, args=args, max_nfev=max_iterations,
                                            bounds=bounds, callback=callback_in_for, jac=get_jacobian)
        if PLOT_HISTORY_LEAST_SQUARES:
            plt.plot(history_costs, label='Sum of least squares coeffs')
            plt.title('Cost history of the least squares fit')
            if history_reg is not None:
                plt.plot(history_reg, label='Regularization term')
                plt.legend()
            plt.show()

        previous_nk = z_surface.nk + (n_images - 1) * 3
        zernike_coeffs_previous = optimization_result.x[:previous_nk].copy()
        zernike_coeffs_visibility = optimization_result.x[previous_nk:previous_nk + z_visibility.nk * n_images].copy()
        zernike_coeffs_brightness = optimization_result.x[previous_nk + z_visibility.nk * n_images:].copy()
    final_rmse = np.sqrt(np.mean(optimization_result.fun**2))
    print_if_verbose(verbose, f"Final RMSE: {final_rmse}")
    zernike_coeffs_surface = zernike_coeffs_previous[3 * (n_images - 1):].copy()

    low_order_surface_coeffs = zernike_coeffs_previous[:3 * n_images].copy()
    zernike_coeffs_surface[:3] = 0.0  # Set piston and tilt coefficients to zero
    fitted_surface = z_surface.eval_grid(zernike_coeffs_surface, matrix=True)
    if plot:
        axs[2].imshow(fitted_surface, cmap="gray")
        axs[2].set_title('Fitted Surface')

        indices_first_interferogram = get_im_zernike_coeffs_indices(0, z_surface, z_visibility, z_brightness, n_images)
        fitted_interferogram = eval_interferogram_model(optimization_result.x[indices_first_interferogram], z_surface,
                                                        z_visibility, z_brightness)
        axs[3].imshow(fitted_interferogram, cmap="gray")
        axs[3].set_title('Fitted Interferogram')
        plt.show()

    return (
        zernike_coeffs_surface, zernike_coeffs_visibility, zernike_coeffs_brightness,
        fitted_surface, final_rmse, low_order_surface_coeffs,
    )


def zern_from_nk(nk):
    order = int((np.sqrt(8 * nk + 1) - 3) / 2)  # Calculate the order from the number of coefficients
    return RZern(order)


def mat_xy(shape, diameter):
    max_x = shape[1] / diameter
    arr_x = np.linspace(-max_x, max_x, shape[1])
    max_y = shape[0] / diameter
    arr_y = np.linspace(-max_y, max_y, shape[0])
    mat_x, mat_y = np.meshgrid(arr_x, arr_y)
    return mat_x, mat_y


def zernike_surface(zernike_coeffs, shape, diameter):
    mat_x, mat_y = mat_xy(shape, diameter)

    z_surface = zern_from_nk(len(zernike_coeffs))
    z_surface.make_cart_grid(mat_x, mat_y)
    fitted_surface = z_surface.eval_grid(zernike_coeffs, matrix=True)
    fitted_surface[np.isnan(fitted_surface)] = 0
    return fitted_surface


def zernike_fit_lsq(order, surface, diameter, output_rmse=False):
    mat_x, mat_y = mat_xy(surface.shape, diameter)

    z_surface = RZern(order)
    z_surface.make_cart_grid(mat_x, mat_y)
    zernike_coeffs = z_surface.fit_cart_grid(surface)[0]
    if output_rmse:
        surface_fitted = z_surface.eval_grid(zernike_coeffs, matrix=True)
        mask = np.logical_not(np.isnan(surface_fitted))
        sum_se = np.sum((surface_fitted[mask] - surface[mask]) ** 2)
        rmse = np.sqrt(sum_se / np.sum(mask))
        return zernike_coeffs, rmse
    return zernike_coeffs


def fit_interferogram_with_zernikes(interferogram, max_order_phase: int=4, max_order_visibility: int=4,
                                    max_order_brightness: int=4, diameter_px: float=None, plot=False, verbose=False,
                                    get_full_model=False):
    """
    Fit a single or multiple interferograms with Zernike polynomials up to a specified order. In the case of multiple
    interferograms, the model uses a single set of Zernike coefficients for the phase for n>1 and a set of coefficients
    for the piston and tilt phase, visibility and brightness for each interferogram.

    Parameters:
    - interferogram: 2D numpy array representing the interferogram or a list of interferograms.
    - max_order_phase: Maximum order of Zernike polynomials for the phase component.
    - max_order_visibility: Maximum order of Zernike polynomials for the visibility component.
    - max_order_brightness: Maximum order of Zernike polynomials for the brightness component.
    - diameter_px: Diameter of the circular aperture in pixels. If None, it will be estimated from the interferogram.

    Returns:
    - zernike_coeffs: Coefficients of the fitted Zernike polynomials.
    - fitted_surface: The surface reconstructed from the fitted Zernike coefficients.
    """
    if isinstance(interferogram, np.ndarray) and np.ndim(interferogram) == 2:
        first_interferogram = interferogram
    else:
        first_interferogram = interferogram[0]
        assert all(interferogram[0].shape == ig.shape for ig in interferogram), (
            'All interferograms must have the same shape.'
        )
    if diameter_px is None:
        # Estimate diameter from the (first) interferogram
        threshold = np.mean(first_interferogram) + 0.5 * np.std(first_interferogram)
        mask = first_interferogram > threshold
        y_indices, x_indices = np.where(mask)
        diameter_px = max(x_indices.max() - x_indices.min(), y_indices.max() - y_indices.min())

    (
        zernike_coeffs_surface, zernike_coeffs_visibility, zernike_coeffs_brightness,
        fitted_surface, final_rmse, low_order_surface_coeffs,
    ) = zernike_fit_interferogram(
        interferogram, max_order_phase, max_order_visibility, max_order_brightness, diameter=diameter_px,
        plot=plot, verbose=verbose, progressive_order_increase=PROGRESSIVE_ORDER_INCREASE
    )

    # Reconstruct surface from Zernike coefficients
    # fitted_surface = zernike_surface(zernike_coeffs_surface, IMAGE_SHAPE, diameter=diameter_px)
    fitted_surface = fitted_surface / (2 * np.pi)
    fitted_surface[np.isnan(fitted_surface)] = 0
    zernike_coeffs_surface = zernike_coeffs_surface / (2 * np.pi)
    low_order_surface_coeffs = low_order_surface_coeffs / (2 * np.pi)

    if get_full_model:
        return (
            zernike_coeffs_surface, fitted_surface, final_rmse, zernike_coeffs_visibility, zernike_coeffs_brightness,
            low_order_surface_coeffs,
        )
    return zernike_coeffs_surface, fitted_surface, final_rmse


def filter_as_zernike(surface, order, diameter_px):
    zern_coeffs = zernike_fit_lsq(order, surface, diameter_px)
    surface = zernike_surface(zern_coeffs, surface.shape, diameter_px)
    return surface, zern_coeffs


def remove_piston_and_tilt_with_zernikes(surface, diameter_px):
    zern_coeffs = zernike_fit_lsq(1, surface, diameter_px)
    surface = surface - zernike_surface(zern_coeffs, surface.shape, diameter_px)
    surface[np.isnan(surface)] = 0
    return surface


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


def test_interferogram_fit():
    from itertools import product

    from tqdm import tqdm

    from FlechaInterfranja.interferogram_generation import FlatInterferogramGenerator

    num_n_fringes = 10
    num_max_deviations = 20
    num_orders = 3
    n_fringes = np.linspace(5, 30, num_n_fringes)
    max_deviations = np.linspace(10, 300, num_max_deviations)
    orders = np.round(np.logspace(np.log10(4), np.log10(9), num_orders)).astype(int)

    generator = FlatInterferogramGenerator(shape=IMAGE_SHAPE, wavelength_nm=WAVELENGTH, pixel_size=PITCH,
                                           min_fringe=N_FRINGES, max_fringe=N_FRINGES, diameter=DIAMETER,
                                           max_rotation=MAX_FRINGES_ROTATION, visibility_ratio=VISIBILITY_RATIO,
                                           seed=RANDOM_SEED)
    rmses = np.zeros((num_n_fringes, num_max_deviations, num_orders))
    prod_enumerate = product(enumerate(n_fringes), enumerate(max_deviations), enumerate(orders))
    total_iterations = num_n_fringes * num_max_deviations * num_orders

    for (i, nfringes), (j, max_deviation), (k, order) in tqdm(prod_enumerate, total=total_iterations):
        frequency = nfringes / generator.diameter_pixels
        generator.current_maximum_deviation_nm = max_deviation
        surface = generator.simulate_surface()
        surface = remove_piston_and_tilt_with_zernikes(surface, generator.diameter_pixels)
        generator.surface = surface
        interferogram = generator.generate_flat_interferogram(normalized_carrier_frequency=frequency)

        (
            zernike_coeffs, fitted_surface, rmse_fit, zc_visibility, zc_brightness,
            low_order_surface_coeffs) = fit_interferogram_with_zernikes(
            interferogram, order, ORDER_VISIBILITY, ORDER_BRIGHTNESS, diameter_px=generator.diameter_pixels,
            verbose=False, get_full_model=True
        )
        zernike_coeffs_direct = zernike_fit_lsq(order, surface, generator.diameter_pixels)
        # Set piston and tilt coefficients to the estimated low order ones
        zernike_coeffs_direct[:3] = low_order_surface_coeffs
        interferogram_direct = eval_interferogram_model_from_coeffs(zernike_coeffs_direct * 2*np.pi, zc_visibility,
                                                                    zc_brightness, IMAGE_SHAPE,
                                                                    generator.diameter_pixels)
        mask = np.logical_not(np.isnan(interferogram_direct))
        rmse_direct = np.sqrt(np.mean((interferogram_direct[mask] - interferogram[mask]) ** 2))

        if PLOT_ALL_RESULTS_IN_TEST:

            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            axs[0].imshow(surface, cmap="gray")
            axs[0].set_title("Original surface")
            axs[1].imshow(fitted_surface, cmap="gray")
            axs[1].set_title(f"Fitted surface with order {order}")
            nk = len(zernike_coeffs)
            axs[2].plot(np.arange(3, nk), zernike_coeffs_direct[3:],
                        label=f"Direct fit of surface (RMSE interferogram: {rmse_direct:.2f})")
            axs[2].plot(np.arange(3, nk), zernike_coeffs[3:],
                        label=f"Fit from interferogram (RMSE interferogram: {rmse_fit:.2f})")
            axs[2].set_title(f"Zernike coefficients for order {order}")
            axs[2].legend()
            plt.show()
        rmse = np.sqrt(np.mean((surface - fitted_surface) ** 2))
        rmses[i, j, k] = rmse

    np.savez("test_interferogram_fit_results.npz", rmses=rmses, n_fringes=n_fringes, max_deviations=max_deviations,
             orders=orders)
    fig, axs = plt.subplots(1, 2, figsize=(15, 8))
    show_order_idx = num_orders // 2
    show_order = orders[show_order_idx]
    axs[0].set_title(f"RMSE for order {show_order}")
    axs[0].imshow(rmses[:, :, show_order_idx], aspect='auto', origin='lower')
    axs[0].set_xlabel("Max deviation (nm)")
    axs[0].set_ylabel("Number of fringes")
    axs[1].set_title(f"RMSE averaged for max deviation {max_deviations[0]:.1f} to {max_deviations[-1]:.1f} nm")
    axs[1].imshow(np.mean(rmses, axis=1), aspect='auto', origin='lower')
    axs[1].set_xlabel("Number of fringes")
    axs[1].set_ylabel("Order of Zernike polynomials")
    plt.show()


def test_multi_interferogram_fit():
    from itertools import product

    from tqdm import tqdm

    from FlechaInterfranja.interferogram_generation import FlatInterferogramGenerator

    num_n_fringes = 10
    num_max_deviations = 10
    num_orders = 3
    n_fringes = np.linspace(5, 30, num_n_fringes)
    max_deviations = np.linspace(10, 300, num_max_deviations)
    orders = np.round(np.logspace(np.log10(4), np.log10(9), num_orders)).astype(int)

    generator = FlatInterferogramGenerator(shape=IMAGE_SHAPE, wavelength_nm=WAVELENGTH, pixel_size=PITCH,
                                           min_fringe=N_FRINGES, max_fringe=N_FRINGES, diameter=DIAMETER,
                                           max_rotation=MAX_FRINGES_ROTATION, visibility_ratio=VISIBILITY_RATIO,
                                           seed=RANDOM_SEED)
    rmses = np.zeros((num_max_deviations, num_orders))
    prod_enumerate = product(enumerate(max_deviations), enumerate(orders))
    total_iterations = num_max_deviations * num_orders

    for (j, max_deviation), (k, order) in tqdm(prod_enumerate, total=total_iterations):
        # Generate a number of interferograms with different number of fringes but the same surface
        generator.current_maximum_deviation_nm = max_deviation
        surface = generator.simulate_surface()
        surface = remove_piston_and_tilt_with_zernikes(surface, generator.diameter_pixels)
        generator.surface = surface
        interferograms = []
        for nfringes in n_fringes:
            frequency = nfringes / generator.diameter_pixels
            interferograms.append(generator.generate_flat_interferogram(normalized_carrier_frequency=frequency,
                                                                        allow_surface_rotation=False))
        (
            zernike_coeffs, fitted_surface, rmse_fit, zc_visibility, zc_brightness,
            low_order_surface_coeffs) = fit_interferogram_with_zernikes(
            interferograms, order, ORDER_VISIBILITY, ORDER_BRIGHTNESS, diameter_px=generator.diameter_pixels,
            verbose=False, get_full_model=True
        )
        zernike_coeffs_direct = zernike_fit_lsq(order, surface, generator.diameter_pixels)
        rmse_directs = np.zeros(num_n_fringes)
        for counter_image in range(num_n_fringes):
            # Set piston and tilt coefficients to the estimated low order ones
            zernike_coeffs_direct[:3] = low_order_surface_coeffs[counter_image * 3:counter_image * 3 + 3]
            interferogram_direct = eval_interferogram_model_from_coeffs(zernike_coeffs_direct * 2*np.pi, zc_visibility,
                                                                        zc_brightness, IMAGE_SHAPE,
                                                                        generator.diameter_pixels)
            mask = np.logical_not(np.isnan(interferogram_direct))
            rmse_directs[counter_image] = np.sqrt(
                np.mean((interferogram_direct[mask] - interferograms[counter_image][mask]) ** 2)
            )

        if PLOT_ALL_RESULTS_IN_TEST:
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            axs[0].imshow(surface, cmap="gray")
            axs[0].set_title("Original surface")
            axs[1].imshow(fitted_surface, cmap="gray")
            axs[1].set_title(f"Fitted surface with order {order}")
            nk = len(zernike_coeffs)
            axs[2].plot(np.arange(3, nk), zernike_coeffs_direct[3:],
                        label=f"Direct fit of surface (RMSE interferogram: {rmse_directs.mean():.2f})")
            axs[2].plot(np.arange(3, nk), zernike_coeffs[3:],
                        label=f"Fit from interferogram (RMSE interferogram: {rmse_fit:.2f})")
            axs[2].set_title(f"Zernike coefficients for order {order}")
            axs[2].legend()
            plt.show()
        rmse = np.sqrt(np.mean((surface - fitted_surface) ** 2))
        rmses[j, k] = rmse

    np.savez("test_interferogram_fit_results.npz", rmses=rmses, n_fringes=n_fringes, max_deviations=max_deviations,
             orders=orders)
    plt.imshow(rmses, aspect='auto', origin='lower',
               extent=[orders[0], orders[-1], max_deviations[0], max_deviations[-1]])
    plt.title("RMSE for different orders and max deviations")
    plt.xlabel("Order of Zernike polynomials (not exact)")
    plt.ylabel("Max deviation (nm)")
    plt.show()


IMAGE_SHAPE = (256, 256)
WAVELENGTH = 632.8  # Wavelength in nm
PITCH = 65e-6  # Pitch in meters
N_FRINGES = 5  # Number of fringes
DIAMETER = 15e-3  # Diameter in meters
MAX_FRINGES_ROTATION = 30.0  # Maximum rotation in degrees of the fringes
VISIBILITY_RATIO = 1.0  # Visibility ratio
MAX_DEVIATION_NM = 100.0  # Maximum deviation in nm

ORDER_PHASE = 6  # Maximum order of Zernike polynomials for phase
ORDER_VISIBILITY = 0  # Maximum order of Zernike polynomials for visibility
ORDER_BRIGHTNESS = 0  # Maximum order of Zernike polynomials for brightness
REGULARIZATION_ALPHA = 2.0

RANDOM_SEED = 0
PLOT_ALL_RESULTS_IN_TEST = False

if __name__ == "__main__":
    test_multi_interferogram_fit()
