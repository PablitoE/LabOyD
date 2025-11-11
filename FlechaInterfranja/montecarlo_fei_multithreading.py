import os
from datetime import datetime
import numpy as np
import flecha_interfranja as fei
from uncertainties.unumpy import nominal_values, uarray
import matplotlib.pyplot as plt
import logging
from multiprocessing import Pool
from interferogram_generation import FlatInterferogramGenerator


logger = logging.getLogger(__name__)
np.random.seed(50)


def worker(simulated_deviation_nm):
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

        interfringe, arrow = fei.analyze_interference(image_array=interferogram, save=False,
                                                      show_result=PLOT_FRINGE_DETECTION)
        arrows.append(arrow)
        interfringes.append(interfringe)
    arrows = nominal_values(arrows)
    interfringes = nominal_values(interfringes)
    simulated_interfringe_spacings = 1 / generator.carrier_frequencies
    measured_interfringe_spacings = interfringes

    pendiente = fei.analisis_flechas_interfranjas(arrows, interfringes, full_output=False)

    if SIMULATION_MODE in ["random", "random_maxfixed"]:
        simulated_deviation_nm = generator.current_maximum_deviation_nm
    measured_max_deviation = pendiente * WAVELENGTH_NM / 2

    return simulated_interfringe_spacings, measured_interfringe_spacings, simulated_deviation_nm, measured_max_deviation


if __name__ == "__main__":
    SAVE_PATH = "Data/Resultados/MonteCarloFEI"
    logging.basicConfig(
        level=logging.CRITICAL,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.FileHandler(os.path.join(SAVE_PATH, "mc_flecha_interfranja.log")),
                  logging.StreamHandler()]
    )

    N_MC_SAMPLES = 10
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

    measured_max_deviations = uarray(np.zeros(N_MC_SAMPLES), np.zeros(N_MC_SAMPLES))
    if SIMULATION_MODE in ["random", "random_maxfixed"]:
        simulated_deviation_nm = np.zeros(N_MC_SAMPLES)
    elif SIMULATION_MODE == "gaussian":
        simulated_deviation_nm = np.linspace(0, MAX_DEVIATION_NM, N_MC_SAMPLES)
    else:
        raise ValueError("SIMULATION_MODE debe ser 'random', 'random_maxfixed' o 'gaussian'")

    measured_interfringe_spacings = np.zeros((N_MC_SAMPLES, N_IMS_PER_SAMPLE))
    simulated_interfringe_spacings = np.zeros((N_MC_SAMPLES, N_IMS_PER_SAMPLE))

    start_time = datetime.now()
    with Pool() as pool:
        results = pool.map(worker, simulated_deviation_nm)
    end_time = datetime.now()
    logging.critical(f"Tiempo de ejecución: {end_time - start_time}")

    for i in range(N_MC_SAMPLES):
        (
            simulated_interfringe_spacings[i, :],
            measured_interfringe_spacings[i, :],
            simulated_deviation_nm[i],
            measured_max_deviations[i],
        ) = results[i]

    errors = np.sqrt(np.abs(nominal_values(measured_max_deviations) - simulated_deviation_nm) ** 2)
    logger.info(f"RMSE: {np.mean(errors)}")
    logger.info(simulated_deviation_nm)
    logger.info(measured_max_deviations)
    date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    np.savez(os.path.join(SAVE_PATH, f"{date}_results.npz"), simulated_deviation_nm=simulated_deviation_nm,
             measured_max_deviations=measured_max_deviations)

    coeffs = np.polyfit(simulated_deviation_nm, nominal_values(measured_max_deviations), 1)
    poly1d_fn = np.poly1d(coeffs)
    logger.info(f"Linear relationship: y = {coeffs[0]:.2f}x + {coeffs[1]:.2f}")
    predicted_measured_deviations = poly1d_fn(simulated_deviation_nm)
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
