import matplotlib.pyplot as plt
import numpy as np
import zernike_fit as zfit
from scipy.ndimage import binary_erosion, rotate

from FlechaInterfranja.interferogram_generation import FlatInterferogramGenerator

IMAGE_SHAPE = (256, 256)
WAVELENGTH = 632.8  # Wavelength in nm
ROTATION_FOURTH_IMAGE_DEG = 45.0  # Rotation of the fourth image in degrees

PITCH = 65e-6  # Pitch in meters
N_FRINGES = 5  # Number of fringes
DIAMETER = 15e-3  # Diameter in meters
MAX_FRINGES_ROTATION = 30.0  # Maximum rotation in degrees of the fringes
VISIBILITY_RATIO = 1.0  # Visibility ratio
MAX_DEVIATION_NM = 100.0  # Maximum deviation in nm

ORDER_PHASE = 6  # Maximum order of Zernike polynomials for phase
ORDER_VISIBILITY = 0  # Maximum order of Zernike polynomials for visibility
ORDER_BRIGHTNESS = 0  # Maximum order of Zernike polynomials for brightness

RANDOM_SEED = 0

PLOT_INTERFEROGRAMS = False
PLOT_MEASURED_SURFACES = False
PLOT_RESULTING_SURFACES = True

SHOW_COEFF_ERRORS = True
DEBUG_FRITZ_MOCK_FIT = False
MAKE_SURFACES_AS_ZERNIKES = True


def fritz_algorithm(z_d, z_e, z_f, z_g, rotation_rad):
    max_coeff = len(z_d)
    zern = zfit.zern_from_nk(max_coeff)
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
    zern = zfit.zern_from_nk(len(zernike_coeffs))
    mirrored_zernike_coeffs = np.copy(zernike_coeffs)
    for i in range(len(zernike_coeffs)):
        n, m = zern.noll2nm(i + 1)
        if (n % 2 == 1 and m > 0) or (n % 2 == 0 and m < 0):
            mirrored_zernike_coeffs[i] = -mirrored_zernike_coeffs[i]
    return mirrored_zernike_coeffs


def rotate_zernike_coeffs(zernike_coeffs, angle_rad):
    zern = zfit.zern_from_nk(len(zernike_coeffs))
    rotated_zernike_coeffs = np.copy(zernike_coeffs)
    for i in range(len(zernike_coeffs)):
        n, m = zern.noll2nm(i + 1)
        neg_m_index = zern.nm2noll(n, -m) - 1
        if m != 0:
            rotated_zernike_coeffs[i] = zernike_coeffs[i] * np.cos(m * angle_rad) + (
                zernike_coeffs[neg_m_index] * np.sin(m * angle_rad))
    return rotated_zernike_coeffs


def plot_compare_ims(list_of_ims_1, names_1, list_of_ims_2, names_2):
    n_ims = len(list_of_ims_1)
    assert n_ims == len(list_of_ims_2)
    assert n_ims == len(names_1)
    assert n_ims == len(names_2)

    fig, axs = plt.subplots(3, n_ims, figsize=(10, 10))
    for i in range(n_ims):
        im1 = list_of_ims_1[i]
        im1[np.isnan(im1)] = 0
        im2 = list_of_ims_2[i]
        im2[np.isnan(im2)] = 0
        name1 = names_1[i]
        name2 = names_2[i]
        vrange = max(abs(np.min(im1)), abs(np.max(im2)), abs(np.min(im2)), abs(np.max(im1)))
        axs[0, i].imshow(im1, cmap='gray', vmin=-vrange, vmax=vrange)
        axs[0, i].set_title(name1)
        axs[1, i].imshow(im2, cmap='gray', vmin=-vrange, vmax=vrange)
        axs[1, i].set_title(name2)
        axs[2, i].imshow(im1 - im2, cmap='gray', vmin=-vrange, vmax=vrange)
        axs[2, i].set_title('Error (1-2)')
    plt.show()

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
    surfaceA = zfit.remove_piston_and_tilt_with_zernikes(surfaceA, generator.diameter_pixels)
    generator.current_maximum_deviation_nm = max_deviationB
    surfaceB = generator.simulate_surface()      # Surface B (l in Fritz algorithm)
    surfaceB = zfit.remove_piston_and_tilt_with_zernikes(surfaceB, generator.diameter_pixels)
    generator.current_maximum_deviation_nm = max_deviationC
    surfaceC = generator.simulate_surface()      # Surface C (m in Fritz algorithm)
    surfaceC = zfit.remove_piston_and_tilt_with_zernikes(surfaceC, generator.diameter_pixels)
    shrunk_mask = generator.aperture_mask.copy()
    shrunk_mask = binary_erosion(shrunk_mask, structure=np.ones((3, 3)), iterations=2)

    surfaceA_filtered, zern_coeffs_A = zfit.filter_as_zernike(surfaceA, ORDER_PHASE, generator.diameter_pixels)
    surfaceB_filtered, zern_coeffs_B = zfit.filter_as_zernike(surfaceB, ORDER_PHASE, generator.diameter_pixels)
    surfaceC_filtered, zern_coeffs_C = zfit.filter_as_zernike(surfaceC, ORDER_PHASE, generator.diameter_pixels)
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
    mirrored_zern_coeffs_B = mirror_x_zernike_coeffs(zern_coeffs_B)
    expected_zern_BA = zern_coeffs_A + mirrored_zern_coeffs_B
    expected_zern_BC = zern_coeffs_C + mirrored_zern_coeffs_B
    expected_zern_AC = zern_coeffs_C + mirror_x_zernike_coeffs(zern_coeffs_A)
    rotated_zern_coeffs_C = rotate_zernike_coeffs(zern_coeffs_C, rotation_fourth_image_rad)
    expected_zern_BCrot = rotated_zern_coeffs_C + mirrored_zern_coeffs_B

    if DEBUG_FRITZ_MOCK_FIT:
        zernike_coeffs_BA = zfit.zernike_fit_lsq(ORDER_PHASE, surface_BA, generator.diameter_pixels)
        zernike_coeffs_BC = zfit.zernike_fit_lsq(ORDER_PHASE, surface_BC, generator.diameter_pixels)
        zernike_coeffs_AC = zfit.zernike_fit_lsq(ORDER_PHASE, surface_AC, generator.diameter_pixels)
        zernike_coeffs_BCrot = zfit.zernike_fit_lsq(ORDER_PHASE, surface_BCrot, generator.diameter_pixels)
    else:
        print(f"Frequencies A-B: {frequencies_BA}")
        zernike_coeffs_BA, fitted_surface_BA = zfit.fit_interferogram_with_zernikes(
            interferogram_BA, ORDER_PHASE, ORDER_VISIBILITY, ORDER_BRIGHTNESS, diameter_px=generator.diameter_pixels
        )
        print(f"{'-'*50}\nFrequencies B-C: {frequencies_BC}")
        zernike_coeffs_BC, fitted_surface_BC = zfit.fit_interferogram_with_zernikes(
            interferogram_BC, ORDER_PHASE, ORDER_VISIBILITY, ORDER_BRIGHTNESS, diameter_px=generator.diameter_pixels
        )
        print(f"{'-'*50}\nFrequencies A-C: {frequencies_AC}")
        zernike_coeffs_AC, fitted_surface_AC = zfit.fit_interferogram_with_zernikes(
            interferogram_AC, ORDER_PHASE, ORDER_VISIBILITY, ORDER_BRIGHTNESS, diameter_px=generator.diameter_pixels
        )
        print(f"{'-'*50}\nFrequencies B-C rotated: {frequencies_BCrot}")
        zernike_coeffs_BCrot, fitted_surface_BCrot = zfit.fit_interferogram_with_zernikes(
            interferogram_BCrot, ORDER_PHASE, ORDER_VISIBILITY, ORDER_BRIGHTNESS, diameter_px=generator.diameter_pixels
        )
        print(f"{'-'*50}")

        if PLOT_MEASURED_SURFACES:
            plot_compare_ims([fitted_surface_BA, fitted_surface_BC, fitted_surface_AC, fitted_surface_BCrot],
                             ['Fitted surface A-B', 'Fitted surface B-C', 'Fitted surface A-C',
                              'Fitted surface B-C rotated'],
                             [surface_BA, surface_BC, surface_AC, surface_BCrot],
                             ['Surface A-B', 'Surface B-C', 'Surface A-C', 'Surface B-C rotated'])

        rmse_surfaces = [np.sqrt(np.mean((surface_BA[shrunk_mask] - fitted_surface_BA[shrunk_mask])**2)),
                        np.sqrt(np.mean((surface_BC[shrunk_mask] - fitted_surface_BC[shrunk_mask])**2)),
                        np.sqrt(np.mean((surface_AC[shrunk_mask] - fitted_surface_AC[shrunk_mask])**2)),
                        np.sqrt(np.mean((surface_BCrot[shrunk_mask] - fitted_surface_BCrot[shrunk_mask])**2))]
        print(f"RMSE surfaces: {np.mean(rmse_surfaces):.2e} half_lambdas")

        if SHOW_COEFF_ERRORS:
            print("Errors in zernike coeffs of the combinations:")
            print("Error in Surface A-B zernike coeffs: ",
                  f"{abs(zernike_coeffs_BA - expected_zern_BA) / abs(expected_zern_BA)}")
            print("Error in Surface B-C zernike coeffs: ",
                  f"{abs(zernike_coeffs_BC - expected_zern_BC) / abs(expected_zern_BC)}")
            print("Error in Surface A-C zernike coeffs: ",
                  f"{abs(zernike_coeffs_AC - expected_zern_AC) / abs(expected_zern_AC)}")
            print("Error in Surface B-C rotated zernike coeffs: ",
                  f"{abs(zernike_coeffs_BCrot - expected_zern_BCrot) / abs(expected_zern_BCrot)}")
            print(f"{'-'*50}")

    z_C, z_A, z_B = fritz_algorithm(zernike_coeffs_BA, zernike_coeffs_BC, zernike_coeffs_BCrot, zernike_coeffs_AC,
                                    rotation_rad=-rotation_fourth_image_rad)

    # Evaluate the results
    if SHOW_COEFF_ERRORS:
        print(f"Error in Surface A zernike coeffs: {abs(zern_coeffs_A - z_A)}")
        print(f"Error in Surface B zernike coeffs: {abs(zern_coeffs_B - z_B)}")
        print(f"Error in Surface C zernike coeffs: {abs(zern_coeffs_C - z_C)}")
        print(f"{'-'*50}")

    result_A = zfit.zernike_surface(z_A, IMAGE_SHAPE, generator.diameter_pixels)
    error_A_nm = (result_A - surfaceA) * WAVELENGTH / 2
    print(f"RMSE A: {np.sqrt(np.mean(error_A_nm[shrunk_mask]**2)):.2f} nm")
    result_B = zfit.zernike_surface(z_B, IMAGE_SHAPE, generator.diameter_pixels)
    error_B_nm = (result_B - surfaceB) * WAVELENGTH / 2
    print(f"RMSE B: {np.sqrt(np.mean(error_B_nm[shrunk_mask]**2)):.2f} nm")
    result_C = zfit.zernike_surface(z_C, IMAGE_SHAPE, generator.diameter_pixels)
    error_C_nm = (result_C - surfaceC) * WAVELENGTH / 2
    print(f"RMSE C: {np.sqrt(np.mean(error_C_nm[shrunk_mask]**2)):.2f} nm")

    if PLOT_RESULTING_SURFACES:
        plot_compare_ims([result_A, result_B, result_C],
                         ['Resulting surface A', 'Resulting surface B', 'Resulting surface C'],
                         [surfaceA, surfaceB, surfaceC],
                         ['Surface A', 'Surface B', 'Surface C'])


    """ Hay que mejorar la estimación de coeficientes de Zernike a partir del interferograma porque el algoritmo de
    Fritz anda bien cuando se mockea la estimación.
    """
