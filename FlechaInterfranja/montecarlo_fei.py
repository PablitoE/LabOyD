import os
import pickle
from datetime import datetime
import numpy as np
import flecha_interfranja as fei
from uncertainties.unumpy import nominal_values, uarray, std_devs
import matplotlib.pyplot as plt
import logging
from multiprocessing import Pool
from interferogram_generation import FlatInterferogramGenerator
from tqdm import tqdm
from Varios.plot import boxplot_by_bins


BASE_SEED = 50
logger = logging.getLogger(__name__)


def worker_star(args):
    return worker(*args)


def worker(sim_id, simulated_deviation_nm, generator=None):
    np.random.seed(BASE_SEED + sim_id)

    if generator is None:
        generator = FlatInterferogramGenerator(shape=(1024, 1024),
                                               pixel_size=65e-6,
                                               save_path=SAVE_PATH,
                                               wavelength_nm=632.8,
                                               min_fringe=MIN_N_FRINGES,
                                               max_fringe=MAX_N_FRINGES,
                                               diameter=50e-3,
                                               max_rotation=MAX_ROTATION_DEG,
                                               visibility_ratio=VISIBILITY_RATIO,
                                               noise_level=NOISE_LEVEL,
                                               maximum_deviation_nm=MAX_DEVIATION_NM
                                               )
    arrows = []
    interfringes = []
    error_rotation_estimation = np.zeros(N_IMS_PER_SAMPLE)
    error_rotation_estimation_corrected = np.zeros(N_IMS_PER_SAMPLE)
    mean_rms_distance_to_valley = np.zeros(N_IMS_PER_SAMPLE)
    simulated_rotation_angle = np.zeros(N_IMS_PER_SAMPLE)
    generator.prepare_next_generation(mode=SIMULATION_MODE, gaussian_max_deviation=simulated_deviation_nm)
    for k_interferogram, interferogram in enumerate(generator.generate(
        num_samples=N_IMS_PER_SAMPLE, output_mode="array", simulation_mode=SIMULATION_MODE,
        surface_options={"plot_surface": PLOT_SURFACES}
    )):
        simulated_rotation_angle[k_interferogram] = generator.current_rotation_angle
        debugging_info = {"valley_curves": generator.minima_curves,
                          "simulated_interfringe_spacing": 1 / generator.current_frequency}
        if not MULTIPROCESSING and PLOT_INTERFEROGRAMS:
            generator.plot_interferogram()

        interfringe, arrow = fei.analyze_interference(image_array=interferogram, save=False,
                                                      show_result=PLOT_FRINGE_DETECTION,
                                                      debugging_info=debugging_info)

        # Debugging
        interfringe_error = interfringe.n - 1 / generator.current_frequency
        if abs(interfringe_error) > 2:
            with open("debug_failed_interfringe.pkl", "ab+") as f:
                data_to_save = {
                    "interferogram": interferogram,
                    "debugging_info": debugging_info
                }
                pickle.dump(data_to_save, f)
                # logging.critical("Interfringe error: %f. STOP. Image saved", interfringe_error)
                # quit()
        # else:
        # logging.critical("Interfringe error: %f", interfringe_error)

        arrows.append(arrow)
        interfringes.append(interfringe)
        error_rotation_estimation[k_interferogram] = (
            generator.current_rotation_angle + debugging_info["rotation_angle_estimated"]
        )
        error_rotation_estimation_corrected[k_interferogram] = (
            generator.current_rotation_angle + debugging_info["rotation_angle_estimated_corrected"]
        )
        mean_rms_distance_to_valley[k_interferogram] = debugging_info["rmsd_to_valley_curves"]
    arrows = nominal_values(arrows)
    measured_interfringe_spacings = nominal_values(interfringes)
    measured_interfringe_spacings_std = std_devs(interfringes)
    simulated_interfringe_spacings = 1 / generator.carrier_frequencies

    if not MULTIPROCESSING and PLOT_FITS:
        pendiente, x, y, descartados, no_descartados = fei.analisis_flechas_interfranjas(
            arrows, measured_interfringe_spacings
        )
        fei.plot_flechas_interfranjas(arrows, measured_interfringe_spacings, save_path=None, iqr_factor=0.75)
        # Los resultados parecen dar bastante ajustados a la recta resultante.
    else:
        pendiente = fei.analisis_flechas_interfranjas(
            arrows,
            measured_interfringe_spacings,
            full_output=False,
            uncertainties=[measured_interfringe_spacings_std, 0.5],
        )

    if SIMULATION_MODE in ["random", "random_maxfixed"]:
        simulated_deviation_nm = generator.current_maximum_deviation_nm
    measured_max_deviation = pendiente * WAVELENGTH_NM / 2

    return (
        simulated_interfringe_spacings, measured_interfringe_spacings,
        simulated_deviation_nm, measured_max_deviation, simulated_rotation_angle,
        error_rotation_estimation, error_rotation_estimation_corrected,
        mean_rms_distance_to_valley, arrows
    )


if __name__ == "__main__":
    MULTIPROCESSING = True
    SAVE_PATH = "Data/Resultados/MonteCarloFEI"
    LOAD_FILENAME = ""
    logging.basicConfig(
        level=logging.CRITICAL,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.FileHandler(os.path.join(SAVE_PATH, "mc_flecha_interfranja.log")),
                  logging.StreamHandler()]
    )

    N_MC_SAMPLES = 200
    N_IMS_PER_SAMPLE = 10
    MIN_N_FRINGES = 12
    MAX_N_FRINGES = 26
    MAX_ROTATION_DEG = 20
    VISIBILITY_RATIO = 0.5
    NOISE_LEVEL = 0.015             # Relative to visibility ratio
    WAVELENGTH_NM = 632.8
    MAX_DEVIATION_NM = 150.0
    PLOT_INTERFEROGRAMS = False
    PLOT_SURFACES = False
    PLOT_FITS = False
    PLOT_FRINGE_DETECTION = False
    PLOT_RESULTS = True
    SIMULATION_MODE = "random"  # "random", "random_maxfixed", "gaussian"

    load_file_path = os.path.join(SAVE_PATH, LOAD_FILENAME)
    if os.path.isfile(load_file_path):
        with np.load(load_file_path, allow_pickle=True) as data:
            simulated_deviation_nm = data["simulated_deviation_nm"]
            measured_max_deviations = data["measured_max_deviations"]
    else:
        date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        measured_max_deviations = uarray(np.zeros(N_MC_SAMPLES), np.zeros(N_MC_SAMPLES))
        if SIMULATION_MODE in ["random", "random_maxfixed"]:
            simulated_deviation_nm = np.zeros(N_MC_SAMPLES)
        elif SIMULATION_MODE == "gaussian":
            simulated_deviation_nm = np.linspace(0, MAX_DEVIATION_NM, N_MC_SAMPLES)
        else:
            raise ValueError("SIMULATION_MODE debe ser 'random', 'random_maxfixed' o 'gaussian'")

        measured_interfringe_spacings = np.zeros((N_MC_SAMPLES, N_IMS_PER_SAMPLE))
        simulated_interfringe_spacings = np.zeros((N_MC_SAMPLES, N_IMS_PER_SAMPLE))
        simulated_rotation = np.zeros((N_MC_SAMPLES, N_IMS_PER_SAMPLE))
        error_rotation_estimation = np.zeros((N_MC_SAMPLES, N_IMS_PER_SAMPLE))
        error_rotation_estimation_corrected = np.zeros((N_MC_SAMPLES, N_IMS_PER_SAMPLE))
        mean_rms_distance_to_valley = np.zeros((N_MC_SAMPLES, N_IMS_PER_SAMPLE))
        arrows_all = np.zeros((N_MC_SAMPLES, N_IMS_PER_SAMPLE))

        if MULTIPROCESSING:
            with Pool() as pool:
                args = [(sim_id, simulated_deviation_nm[sim_id]) for sim_id in range(N_MC_SAMPLES)]
                results = list(tqdm(pool.imap(worker_star, args), total=N_MC_SAMPLES))

            for i in range(N_MC_SAMPLES):
                (
                    simulated_interfringe_spacings[i, :], measured_interfringe_spacings[i, :],
                    simulated_deviation_nm[i], measured_max_deviations[i], simulated_rotation[i, :],
                    error_rotation_estimation[i, :], error_rotation_estimation_corrected[i, :],
                    mean_rms_distance_to_valley[i, :], arrows_all[i, :]
                ) = results[i]
        else:
            generator = FlatInterferogramGenerator(shape=(1024, 1024),
                                                   pixel_size=65e-6,
                                                   save_path=SAVE_PATH,
                                                   wavelength_nm=632.8,
                                                   min_fringe=MIN_N_FRINGES,
                                                   max_fringe=MAX_N_FRINGES,
                                                   diameter=50e-3,
                                                   max_rotation=MAX_ROTATION_DEG,
                                                   visibility_ratio=VISIBILITY_RATIO,
                                                   noise_level=NOISE_LEVEL,
                                                   maximum_deviation_nm=MAX_DEVIATION_NM
                                                   )
            for i in tqdm(range(N_MC_SAMPLES), desc='Simulaciones Flecha-Interfranja', total=N_MC_SAMPLES):
                (
                    simulated_interfringe_spacings[i, :], measured_interfringe_spacings[i, :],
                    simulated_deviation_nm[i], measured_max_deviations[i], simulated_rotation[i, :],
                    error_rotation_estimation[i, :], error_rotation_estimation_corrected[i, :],
                    mean_rms_distance_to_valley[i, :], arrows_all[i, :]
                ) = worker(i, simulated_deviation_nm[i], generator=generator)

        if PLOT_RESULTS:
            error_interfringe_spacings = np.abs(measured_interfringe_spacings - simulated_interfringe_spacings)
            interfringe_rmse = np.sqrt(np.mean(error_interfringe_spacings**2, axis=0))
            relative_interfringe_rmse = interfringe_rmse / np.mean(simulated_interfringe_spacings, axis=0)
            fig_interfranja, axs = plt.subplots(2, 2, figsize=(16, 10))
            axs[0, 0].plot(range(1, N_IMS_PER_SAMPLE + 1), interfringe_rmse, 'o-')
            axs[0, 1].plot(range(1, N_IMS_PER_SAMPLE + 1), relative_interfringe_rmse, 'o-')
            axs[0, 0].set_xlabel(f'Número de interferograma ({MIN_N_FRINGES} a {MAX_N_FRINGES} franjas)')
            axs[0, 1].set_xlabel(f'Número de interferograma ({MIN_N_FRINGES} a {MAX_N_FRINGES} franjas)')
            axs[0, 0].set_ylabel('RMSE del espaciamiento interfranja (píxeles)')
            axs[0, 1].set_ylabel('RMSE del espaciamiento interfranja relativo al valor simulado')
            rmse_interfringe_spacings = np.sqrt(np.mean(error_interfringe_spacings**2, axis=1))
            axs[1, 0].plot(simulated_deviation_nm, rmse_interfringe_spacings, 'o')
            axs[1, 0].set_xlabel('Desviación máxima simulada (nm)')
            axs[1, 0].set_ylabel('RMSE del espaciamiento interfranja (píxeles) para todos los espaciamientos')
            relative_error_interfringe = error_interfringe_spacings / simulated_interfringe_spacings
            boxplot_by_bins(simulated_rotation.flatten(), relative_error_interfringe.flatten(), ax=axs[1, 1])
            axs[1, 1].set_xlabel('Rotación simulada (°)')
            axs[1, 1].set_ylabel('Error relativo de espaciamiento interfranja (píxeles)')
            for ax in axs.flatten():
                ax.grid(True)
            fig_interfranja.savefig(os.path.join(SAVE_PATH, f"{date}_interfranja_rmse.png"))

            fig_error_rot_valles, axs = plt.subplots(2, 3, figsize=(16, 10))
            axs[0, 0].boxplot(np.abs(error_rotation_estimation_corrected), positions=range(1, N_IMS_PER_SAMPLE + 1))
            axs[0, 0].set_xlabel(f'Número de interferograma ({MIN_N_FRINGES} a {MAX_N_FRINGES} franjas)')
            axs[0, 0].set_ylabel('Errores en la estimación de rotación (°)')
            rmse_error_rotation = np.sqrt(np.mean(error_rotation_estimation ** 2, axis=1))
            rmse_error_rotation_corrected = np.sqrt(np.mean(error_rotation_estimation_corrected ** 2, axis=1))
            axs[0, 1].plot(
                simulated_deviation_nm, rmse_error_rotation, 'o', label="Rotation by contrast on averaged columns"
            )
            axs[0, 1].plot(
                simulated_deviation_nm, rmse_error_rotation_corrected, 'o',
                label="Corrected rotation by optimizing fringe location"
            )
            axs[0, 1].legend()
            axs[0, 1].set_xlabel('Desviación máxima simulada (nm)')
            axs[0, 1].set_ylabel('RMSE de la estimación de rotación (°)')
            boxplot_by_bins(
                simulated_rotation.flatten(), np.abs(error_rotation_estimation_corrected).flatten(), ax=axs[0, 2]
            )
            axs[0, 2].set_xlabel('Rotación simulada (°)')
            axs[0, 2].set_ylabel('Errores en la estimación de rotación por optimización (°)')

            axs[1, 0].boxplot(mean_rms_distance_to_valley, positions=range(1, N_IMS_PER_SAMPLE + 1))
            axs[1, 0].set_xlabel(f'Número de interferograma ({MIN_N_FRINGES} a {MAX_N_FRINGES} franjas)')
            axs[1, 0].set_ylabel('RMS de la distancia a curvas valle (píxeles)')
            rmse_mean_rms_distance_to_valley = np.sqrt(np.mean(mean_rms_distance_to_valley ** 2, axis=1))
            axs[1, 1].plot(simulated_deviation_nm, rmse_mean_rms_distance_to_valley, 'o')
            axs[1, 1].set_xlabel('Desviación máxima simulada (nm)')
            axs[1, 1].set_ylabel('RMSE de la distancia a curvas valle (píxeles)')
            boxplot_by_bins(simulated_rotation.flatten(), mean_rms_distance_to_valley.flatten(), ax=axs[1, 2])
            axs[1, 2].set_xlabel('Rotación simulada (°)')
            axs[1, 2].set_ylabel('RMSE de la distancia a curvas valle (píxeles)')
            for ax in axs.flatten():
                ax.grid(True)
            fig_error_rot_valles.savefig(os.path.join(SAVE_PATH, f"{date}_error_rotacion_valles.png"))

            fig_error_arrows, axs = plt.subplots(1, 2, figsize=(16, 6))
            simulated_arrows_px = (
                simulated_deviation_nm[:, np.newaxis] / WAVELENGTH_NM * 2 * simulated_interfringe_spacings
            )
            error_arrows = arrows_all - simulated_arrows_px
            error_arrows_rmse_by_interfringe = np.sqrt(np.mean(error_arrows**2, axis=0))
            error_arrows_rel = error_arrows / simulated_arrows_px
            axs[0].plot(range(1, N_IMS_PER_SAMPLE + 1), error_arrows_rmse_by_interfringe, 'o-')
            axs[0].boxplot(error_arrows, positions=range(1, N_IMS_PER_SAMPLE + 1))
            axs[0].set_xlabel(f'Número de interferograma ({MIN_N_FRINGES} a {MAX_N_FRINGES} franjas)')
            axs[0].set_ylabel('Error en la estimación de flecha (píxeles)')
            axs[1].plot(simulated_deviation_nm, np.sqrt(np.mean(error_arrows_rel**2, axis=1)), 'o')
            axs[1].set_xlabel('Desviación máxima simulada (nm)')
            axs[1].set_ylabel('RMSE en la estimación de flecha (relativo a la flecha simulada)')
            for ax in axs:
                ax.grid(True)
            fig_error_arrows.savefig(os.path.join(SAVE_PATH, f"{date}_error_flechas.png"))

        errors = np.sqrt(np.abs(nominal_values(measured_max_deviations) - simulated_deviation_nm) ** 2)
        logger.info(f"RMSE: {np.mean(errors)}")
        # logger.info(simulated_deviation_nm)
        # logger.info(measured_max_deviations)
        np.savez(os.path.join(SAVE_PATH, f"{date}_results.npz"), simulated_deviation_nm=simulated_deviation_nm,
                 measured_max_deviations=measured_max_deviations)

    coeffs = np.polyfit(simulated_deviation_nm, nominal_values(measured_max_deviations), 1)
    proportionality_constant = np.sum(simulated_deviation_nm * nominal_values(measured_max_deviations)) / np.sum(
        simulated_deviation_nm**2
    )
    predicted_measured_deviations = proportionality_constant * simulated_deviation_nm

    errors = np.sqrt(np.abs(nominal_values(measured_max_deviations) - predicted_measured_deviations) ** 2)
    logger.info(f"RMSE (lineal): {np.mean(errors)}")

    if PLOT_RESULTS:
        fig, axs = plt.subplots(1, 3, figsize=(16, 6))
        axs[0].plot(simulated_deviation_nm, nominal_values(measured_max_deviations), 'o')
        axs[0].plot(
            simulated_deviation_nm, predicted_measured_deviations,
            '--', label=f'Ajuste lineal: m={proportionality_constant:.3f}',
        )
        axs[0].set_xlabel('Desviación máxima simulada (nm)')
        axs[0].set_ylabel('Desviación máxima medida (nm)')
        axs[0].grid(True)
        axs[0].legend()
        boxplot_by_bins(
            simulated_deviation_nm, np.abs(
                nominal_values(measured_max_deviations) - predicted_measured_deviations
            ), ax=axs[1]
        )
        axs[1].set_xlabel('Desviación máxima simulada (nm)')
        axs[1].set_ylabel('Errores absolutos contra ajuste lineal (nm)')
        axs[2].plot(simulated_deviation_nm, std_devs(measured_max_deviations), 'o')
        axs[2].set_xlabel('Desviación máxima simulada (nm)')
        axs[2].set_ylabel('Incertidumbre del ajuste proporcional flecha-interfranja (nm)')
        axs[2].grid(True)
        fig.savefig(os.path.join(SAVE_PATH, f"{date}_scatter_desviaciones_maximas.png"))
        plt.show()
