import os
import numpy as np
from scipy.ndimage import rotate, gaussian_filter
from PIL import Image
import flecha_interfranja as fei
from uncertainties.unumpy import nominal_values, uarray
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm


logger = logging.getLogger(__name__)


class FlatInterferogramGenerator():
    def __init__(self, shape=(1024, 1024), pixel_size=65e-6, save_path=None, wavelength_nm=632.8, min_fringe=5,
                 max_fringe=20, diameter=50e-3, max_rotation=0.0, visibility_ratio=1.0, noise_level=0.01,
                 maximum_deviation_nm=70.0):
        assert isinstance(shape, tuple) and len(shape) == 2, "shape debe ser una tupla de dos elementos"
        assert isinstance(pixel_size, (int, float)) and pixel_size > 0, "pixel_size debe ser un número positivo"
        assert isinstance(save_path, str) and save_path != "", "save_path debe ser una cadena no vacía"
        assert isinstance(wavelength_nm, (int, float)) and wavelength_nm > 0, \
               "wavelength_nm debe ser un número positivo"
        assert isinstance(min_fringe, (int, float)) and min_fringe > 0, "min_fringe debe ser un número no negativo"
        assert isinstance(max_fringe, (int, float)) and max_fringe > min_fringe, \
               "max_fringe debe ser un número positivo mayor que min_fringe"
        assert isinstance(diameter, (int, float)) and diameter > 0, "diameter debe ser un número positivo"
        assert isinstance(max_rotation, (int, float)) and 0 <= max_rotation <= 90, \
               "max_rotation debe estar en el rango [0, 90 grados]"
        assert isinstance(visibility_ratio, (int, float)) and 0 < visibility_ratio <= 1.0, \
               "visibility_ratio debe estar en el rango (0, 1]"
        assert isinstance(noise_level, (int, float)) and noise_level >= 0, \
               "noise_level debe ser un número no negativo"
        assert isinstance(maximum_deviation_nm, (int, float)) and maximum_deviation_nm >= 0, \
               "maximum_deviation_nm debe ser un número no negativo"
        self.shape = shape
        self.size = shape[0] * shape[1]
        self.pixel_size = pixel_size
        self.save_path = save_path
        self.wavelength_nm = wavelength_nm
        self.min_fringe = min_fringe
        self.max_fringe = max_fringe
        self.diameter = diameter
        self.max_rotation = max_rotation
        self.visibility_ratio = visibility_ratio
        self.noise_level = noise_level
        self.maximum_deviation_nm = maximum_deviation_nm
        os.makedirs(self.save_path, exist_ok=True)

        X, Y = np.meshgrid(np.arange(self.shape[1]), np.arange(self.shape[0]))
        self.X = X - self.shape[1] // 2
        self.Y = Y - self.shape[0] // 2
        self.r = np.sqrt(self.X**2 + self.Y**2)
        self.aperture_mask = self.r <= (self.diameter_pixels / 2)

        self.surface = None
        self.carrier_frequencies = None

    @property
    def diameter_pixels(self):
        return self.diameter / self.pixel_size

    def produce_carrier_frequencies(self, num_samples):
        if num_samples == 1:
            n_fringes = np.array([(self.min_fringe + self.max_fringe) / 2])
            step_n_fringes = 2 * (self.max_fringe - self.min_fringe)
        else:
            n_fringes = np.linspace(self.min_fringe, self.max_fringe, num_samples)
            step_n_fringes = (self.max_fringe - self.min_fringe) / (num_samples - 1)
        n_fringes = n_fringes + np.random.uniform(-0.25, 0.25, size=num_samples) * step_n_fringes
        n_fringes[n_fringes < 1] = 1.0

        periods_px = self.diameter_pixels / n_fringes
        self.carrier_frequencies = 1 / periods_px

    def random_rotation(self, interferogram):
        if self.max_rotation == 0.0:
            return interferogram

        angle = np.random.uniform(-self.max_rotation, self.max_rotation)

        return rotate(interferogram, angle, reshape=False, order=3, mode='nearest')

    def simulate_surface(self, plot_surface=False, mode="random"):
        if mode == "random":
            surface = np.random.normal(0, 1, size=self.shape)
            surface = gaussian_filter(surface, sigma=np.sqrt(self.size) / 8)
        elif mode == "gaussian":
            sigma = self.diameter_pixels / 3
            surface = np.exp(-(self.r**2) / (2 * sigma**2))
            surface = surface - np.mean(surface)
        surface *= self.aperture_mask
        surface = surface / (np.max(np.abs(surface)) - np.min(np.abs(surface)))
        self.surface = surface * self.current_maximum_deviation_nm / self.wavelength_nm * 2

        if plot_surface:
            plt.imshow(self.surface * self.aperture_mask, cmap='jet')
            plt.colorbar(label='Desviación normalizada (λ unidades)')
            plt.title('Superficie simulada')
            plt.xlabel('Píxeles')
            plt.ylabel('Píxeles')
            plt.show()

    def generate_flat_interferogram(self, normalized_carrier_frequency=0.1):
        kx = normalized_carrier_frequency
        ky = 0.0

        phase = 2 * np.pi * (kx * self.X + ky * self.Y + self.surface)
        interferogram = 1 + self.visibility_ratio * np.cos(phase)
        interferogram *= self.aperture_mask
        interferogram = self.random_rotation(interferogram)
        interferogram += np.random.normal(0, 1, size=self.shape) * self.noise_level
        interferogram /= np.max(interferogram)
        interferogram_uint8 = np.clip(interferogram * 255, 0, 255).astype(np.uint8)

        return interferogram_uint8

    def generate(self, num_samples=1, output_mode="files", simulation_mode="random", surface_options={}):
        assert isinstance(num_samples, int) and num_samples > 0, "num_samples debe ser un entero positivo"
        self.produce_carrier_frequencies(num_samples)
        if simulation_mode == "random":
            self.current_maximum_deviation_nm = np.random.uniform(0, self.maximum_deviation_nm)
            surface_mode = "random"
        elif simulation_mode == "random_maxfixed":
            self.current_maximum_deviation_nm = self.maximum_deviation_nm
            surface_mode = "random"
        elif simulation_mode == "gaussian":
            self.current_maximum_deviation_nm = self.maximum_deviation_nm
            surface_mode = "gaussian"

        self.simulate_surface(mode=surface_mode, **surface_options)

        for i in range(num_samples):
            this_frequency = self.carrier_frequencies[i]
            interferogram = self.generate_flat_interferogram(this_frequency)
            if output_mode == "files":
                filename = f"flat_interferogram_{i+1:03d}.png"
                filepath = os.path.join(self.save_path, filename)
                Image.fromarray(interferogram).save(filepath)
            elif output_mode == "array":
                yield interferogram

    def prepare_next_generation(self, mode="random", gaussian_max_deviation=None):
        if mode == "gaussian":
            generator.maximum_deviation_nm = gaussian_max_deviation


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.CRITICAL,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.FileHandler('mc_flecha_interfranja.log'),
                  logging.StreamHandler()]
    )

    SAVE_PATH = "Data/Resultados/MonteCarloFEI"
    N_MC_SAMPLES = 20
    N_IMS_PER_SAMPLE = 10
    MIN_N_FRINGES = 7
    MAX_N_FRINGES = 20
    MAX_ROTATION_DEG = 1
    VISIBILITY_RATIO = 0.5
    NOISE_LEVEL = 0.0
    WAVELENGTH_NM = 632.8
    MAX_DEVIATION_NM = 50.0
    PLOT_INTERFEROGRAMS = False
    PLOT_SURFACES = False
    PLOT_FITS = False
    PLOT_FRINGE_DETECTION = False
    SIMULATION_MODE = "gaussian"  # "random", "random_maxfixed", "gaussian"
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

    measured_max_deviations = uarray(np.zeros(N_MC_SAMPLES), np.zeros(N_MC_SAMPLES))
    if SIMULATION_MODE in ["random", "random_maxfixed"]:
        simulated_deviation_nm = np.zeros(N_MC_SAMPLES)
    elif SIMULATION_MODE == "gaussian":
        simulated_deviation_nm = np.linspace(0, MAX_DEVIATION_NM, N_MC_SAMPLES)
    else:
        raise ValueError("SIMULATION_MODE debe ser 'random', 'random_maxfixed' o 'gaussian'")

    measured_interfringe_spacings = np.zeros((N_MC_SAMPLES, N_IMS_PER_SAMPLE))
    simulated_interfringe_spacings = np.zeros((N_MC_SAMPLES, N_IMS_PER_SAMPLE))

    for i in tqdm(range(N_MC_SAMPLES), desc='Simulaciones Flecha-Interfranja', total=N_MC_SAMPLES):
        arrows = []
        interfringes = []
        generator.prepare_next_generation(mode=SIMULATION_MODE, gaussian_max_deviation=simulated_deviation_nm[i])
        for interferogram in generator.generate(
            num_samples=N_IMS_PER_SAMPLE, output_mode="array", simulation_mode=SIMULATION_MODE,
            surface_options={"plot_surface": PLOT_SURFACES}
        ):
            if PLOT_INTERFEROGRAMS:
                plt.imshow(interferogram, cmap='gray')
                plt.title('Interferograma simulado (max_dev = %.2f nm)' % generator.current_maximum_deviation_nm)
                plt.axis('off')
                plt.show()

            interfringe, arrow = fei.analyze_interference(image_array=interferogram, save=False,
                                                          show_result=PLOT_FRINGE_DETECTION)
            arrows.append(arrow)
            interfringes.append(interfringe)
        arrows = nominal_values(arrows)
        interfringes = nominal_values(interfringes)
        simulated_interfringe_spacings[i, :] = 1 / generator.carrier_frequencies
        measured_interfringe_spacings[i, :] = interfringes

        if PLOT_FITS:
            pendiente, x, y, descartados, no_descartados = fei.analisis_flechas_interfranjas(arrows, interfringes)
            fei.plot_flechas_interfranjas(arrows, interfringes, save_path=None, iqr_factor=0.75)
            # Los resultados parecen dar bastante ajustados a la recta resultante.
        else:
            pendiente = fei.analisis_flechas_interfranjas(arrows, interfringes, full_output=False)

        if SIMULATION_MODE in ["random", "random_maxfixed"]:
            simulated_deviation_nm[i] = generator.current_maximum_deviation_nm
        measured_max_deviations[i] = pendiente * WAVELENGTH_NM / 2

    mean_interfringe_rmse = np.sqrt(
        np.mean((measured_interfringe_spacings - simulated_interfringe_spacings) ** 2, axis=0)
    )
    _, axs = plt.subplots(1, 2)
    axs[0].plot(range(1, N_IMS_PER_SAMPLE + 1), mean_interfringe_rmse, 'o-')
    axs[1].plot(
        range(1, N_IMS_PER_SAMPLE + 1), mean_interfringe_rmse / np.mean(simulated_interfringe_spacings, axis=0), 'o-'
    )
    axs[0].set_xlabel('Número de interferograma')
    axs[1].set_xlabel('Número de interferograma')
    axs[0].set_ylabel('RMSE del espaciamiento interfranja (píxeles)')
    axs[1].set_ylabel('RMSE del espaciamiento interfranja relativo al valor simulado')
    axs[0].grid(True)
    axs[1].grid(True)
    plt.show()

    errors = np.sqrt(np.abs(nominal_values(measured_max_deviations) - simulated_deviation_nm) ** 2)
    print(f"RMSE: {np.mean(errors)}")
    print(simulated_deviation_nm)
    print(measured_max_deviations)

    if SIMULATION_MODE == "gaussian":
        plt.plot(simulated_deviation_nm, measured_max_deviations, 'o--')
        plt.xlabel('Desviación máxima simulada (nm)')
        plt.ylabel('Desviación máxima medida (nm)')
        plt.grid(True)
        plt.show()
