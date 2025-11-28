import os
from datetime import datetime
import numpy as np
import flecha_interfranja as fei
from uncertainties.unumpy import nominal_values, uarray
import matplotlib.pyplot as plt
import logging
from multiprocessing import Pool
from interferogram_generation import FlatInterferogramGenerator
from tqdm import tqdm


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
    generator.prepare_next_generation(mode=SIMULATION_MODE, gaussian_max_deviation=simulated_deviation_nm)
    for interferogram in generator.generate(
        num_samples=N_IMS_PER_SAMPLE, output_mode="array", simulation_mode=SIMULATION_MODE,
        surface_options={"plot_surface": PLOT_SURFACES}
    ):
        if MULTIPROCESSING:
            debugging_info = None
        else:
            if PLOT_INTERFEROGRAMS:
                plt.imshow(interferogram, cmap='gray')
                plt.title('Interferograma simulado (max_dev = %.2f nm)' % generator.current_maximum_deviation_nm)
                plt.axis('off')
                plt.show()

            debugging_info = {
                "rotation_angle": generator.current_rotation_angle,
                "carrier_frequency": generator.current_frequency,
                "maximum_deviation": generator.current_maximum_deviation_nm
            }

        interfringe, arrow = fei.analyze_interference(image_array=interferogram, save=False,
                                                      show_result=PLOT_FRINGE_DETECTION,
                                                      debugging_info=debugging_info)
        arrows.append(arrow)
        interfringes.append(interfringe)
    arrows = nominal_values(arrows)
    interfringes = nominal_values(interfringes)
    simulated_interfringe_spacings = 1 / generator.carrier_frequencies
    measured_interfringe_spacings = interfringes

    if not MULTIPROCESSING and PLOT_FITS:
        pendiente, x, y, descartados, no_descartados = fei.analisis_flechas_interfranjas(arrows, interfringes)
        fei.plot_flechas_interfranjas(arrows, interfringes, save_path=None, iqr_factor=0.75)
        # Los resultados parecen dar bastante ajustados a la recta resultante.
    else:
        pendiente = fei.analisis_flechas_interfranjas(arrows, interfringes, full_output=False)

    if SIMULATION_MODE in ["random", "random_maxfixed"]:
        simulated_deviation_nm = generator.current_maximum_deviation_nm
    measured_max_deviation = pendiente * WAVELENGTH_NM / 2

    return simulated_interfringe_spacings, measured_interfringe_spacings, simulated_deviation_nm, measured_max_deviation


if __name__ == "__main__":
    MULTIPROCESSING = False
    SAVE_PATH = "Data/Resultados/MonteCarloFEI"
    LOAD_FILENAME = ""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.FileHandler(os.path.join(SAVE_PATH, "mc_flecha_interfranja.log")),
                  logging.StreamHandler()]
    )

    N_MC_SAMPLES = 1
    N_IMS_PER_SAMPLE = 10
    MIN_N_FRINGES = 7
    MAX_N_FRINGES = 20
    MAX_ROTATION_DEG = 1
    VISIBILITY_RATIO = 0.5
    NOISE_LEVEL = 0.0
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
        measured_max_deviations = uarray(np.zeros(N_MC_SAMPLES), np.zeros(N_MC_SAMPLES))
        if SIMULATION_MODE in ["random", "random_maxfixed"]:
            simulated_deviation_nm = np.zeros(N_MC_SAMPLES)
        elif SIMULATION_MODE == "gaussian":
            simulated_deviation_nm = np.linspace(0, MAX_DEVIATION_NM, N_MC_SAMPLES)
        else:
            raise ValueError("SIMULATION_MODE debe ser 'random', 'random_maxfixed' o 'gaussian'")

        measured_interfringe_spacings = np.zeros((N_MC_SAMPLES, N_IMS_PER_SAMPLE))
        simulated_interfringe_spacings = np.zeros((N_MC_SAMPLES, N_IMS_PER_SAMPLE))

        if MULTIPROCESSING:
            start_time = datetime.now()
            with Pool() as pool:
                args = [(sim_id, simulated_deviation_nm[sim_id]) for sim_id in range(N_MC_SAMPLES)]
                results = list(tqdm(pool.imap(worker_star, args), total=N_MC_SAMPLES))
            end_time = datetime.now()
            logging.critical(f"Tiempo de ejecución: {end_time - start_time}")

            for i in range(N_MC_SAMPLES):
                (
                    simulated_interfringe_spacings[i, :],
                    measured_interfringe_spacings[i, :],
                    simulated_deviation_nm[i],
                    measured_max_deviations[i],
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
                    simulated_interfringe_spacings[i, :],
                    measured_interfringe_spacings[i, :],
                    simulated_deviation_nm[i],
                    measured_max_deviations[i],
                ) = worker(i, simulated_deviation_nm[i], generator=generator)

        if PLOT_RESULTS:
            mean_interfringe_rmse = np.sqrt(
                np.mean((measured_interfringe_spacings - simulated_interfringe_spacings) ** 2, axis=0)
            )
            _, axs = plt.subplots(1, 3)
            axs[0].plot(range(1, N_IMS_PER_SAMPLE + 1), mean_interfringe_rmse, 'o-')
            axs[1].plot(
                range(1, N_IMS_PER_SAMPLE + 1), mean_interfringe_rmse / np.mean(simulated_interfringe_spacings, axis=0),
                'o-'
            )
            axs[0].set_xlabel('Número de interferograma')
            axs[1].set_xlabel('Número de interferograma')
            axs[0].set_ylabel('RMSE del espaciamiento interfranja (píxeles)')
            axs[1].set_ylabel('RMSE del espaciamiento interfranja relativo al valor simulado')
            axs[2].plot(
                simulated_deviation_nm,
                np.sqrt(np.mean((measured_interfringe_spacings - simulated_interfringe_spacings) ** 2)),
                'o',
            )
            axs[2].set_xlabel('Desviación máxima simulada (nm)')
            axs[2].set_ylabel('RMSE del espaciamiento interfranja (píxeles)')
            for ax in axs:
                ax.grid(True)
            plt.savefig(os.path.join(SAVE_PATH, "interfranja_rmse.png"))

        errors = np.sqrt(np.abs(nominal_values(measured_max_deviations) - simulated_deviation_nm) ** 2)
        logger.info(f"RMSE: {np.mean(errors)}")
        # logger.info(simulated_deviation_nm)
        # logger.info(measured_max_deviations)
        date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
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
        plt.figure(figsize=(8, 6))
        plt.plot(simulated_deviation_nm, nominal_values(measured_max_deviations), 'o')
        plt.plot(simulated_deviation_nm, predicted_measured_deviations, 'o-', label='Ajuste lineal')
        plt.xlabel('Desviación máxima simulada (nm)')
        plt.ylabel('Desviación máxima medida (nm)')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(SAVE_PATH, "scatter_desviaciones_maximas.png"))
        plt.show()
