import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import rotate, gaussian_filter
from PIL import Image
from skimage import measure
from Varios.images import all_pixels_inside_border, minimum_point_per_row
from Varios.lines_points import rotate_2d_points


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
        self.current_frequency = None
        self.current_maximum_deviation_nm = None
        self.current_rotation_angle = None
        self.minima_curves = None
        self.current_interferogram = None

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

        self.current_rotation_angle = np.random.uniform(-self.max_rotation, self.max_rotation)

        return rotate(interferogram, self.current_rotation_angle, reshape=False, order=3, mode='nearest')

    def simulate_surface(self, no_tilt=True, plot_surface=False, mode="random", choice_ratio_fit_tilt=0.1):
        if mode == "random":
            surface = np.random.normal(0, 1, size=self.shape)
            surface = gaussian_filter(surface, sigma=np.sqrt(self.size) / 8)
        elif mode == "gaussian":
            sigma = self.diameter_pixels / 3
            surface = np.exp(-(self.r**2) / (2 * sigma**2))
            surface = surface - np.mean(surface)

        if no_tilt:
            num_pixels_to_keep = int(np.sum(self.aperture_mask) * choice_ratio_fit_tilt)
            indices_in_mask = np.argwhere(self.aperture_mask)
            indices_to_keep = np.random.choice(len(indices_in_mask), num_pixels_to_keep, replace=False)
            indices_to_keep = indices_in_mask[indices_to_keep]

            Y_indices = np.arange(self.shape[0])[:, np.newaxis]
            X_indices = np.arange(self.shape[1])[np.newaxis, :]
            A = np.c_[indices_to_keep[:, 1], indices_to_keep[:, 0], np.ones(num_pixels_to_keep)]
            b = surface[indices_to_keep[:, 0], indices_to_keep[:, 1]]
            coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            plane = (coeffs[0] * X_indices + coeffs[1] * Y_indices + coeffs[2])
            surface = surface - plane

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

    def get_maximum_simulated_deviation_px(self, phase, plot=False):
        phase[np.logical_not(self.aperture_mask)] = 0
        cos_phase = 1 + np.cos(phase)
        contours = measure.find_contours(cos_phase, 0.01)
        self.minima_curves = []
        if plot:
            plt.imshow(cos_phase, cmap='jet')
        for contour in contours:
            pixels_inside = all_pixels_inside_border(contour, self.shape)
            self.minima_curves.append(
                minimum_point_per_row(pixels_inside, cos_phase[pixels_inside[:, 0], pixels_inside[:, 1]])
            )
            if plot:
                plt.plot(contour[:, 1], contour[:, 0], linewidth=2)
                plt.plot(self.minima_curves[-1][:, 1], self.minima_curves[-1][:, 0], 'r--', linewidth=2)
        if plot:
            plt.title('Contornos de interferencia')
            plt.xlabel('Píxeles')
            plt.ylabel('Píxeles')
            plt.colorbar(label='Intensidad')
            plt.show()
        return

    def rotate_simulated_minima_curves(self):
        self.minima_curves = rotate_2d_points(self.minima_curves, -self.current_rotation_angle, self.shape)

    def generate_flat_interferogram(self, normalized_carrier_frequency=0.1):
        kx = normalized_carrier_frequency
        ky = 0.0

        phase = 2 * np.pi * (kx * self.X + ky * self.Y + self.surface)
        self.get_maximum_simulated_deviation_px(phase, plot=False)
        interferogram = 1 + self.visibility_ratio * np.cos(phase)
        interferogram *= self.aperture_mask
        interferogram = self.random_rotation(interferogram)
        interferogram += np.random.normal(0, 1, size=self.shape) * self.noise_level
        interferogram /= np.max(interferogram)
        interferogram_uint8 = np.clip(interferogram * 255, 0, 255).astype(np.uint8)

        self.current_interferogram = interferogram_uint8
        self.rotate_simulated_minima_curves()

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
            self.current_frequency = self.carrier_frequencies[i]
            interferogram = self.generate_flat_interferogram(self.current_frequency)
            if output_mode == "files":
                filename = f"flat_interferogram_{i+1:03d}.png"
                filepath = os.path.join(self.save_path, filename)
                Image.fromarray(interferogram).save(filepath)
            elif output_mode == "array":
                yield interferogram

    def prepare_next_generation(self, mode="random", gaussian_max_deviation=None):
        if mode == "gaussian":
            self.maximum_deviation_nm = gaussian_max_deviation

    def plot_interferogram(self):
        if self.current_interferogram is None:
            raise ValueError("There is no generated interferogram to display.")
        _, axs = plt.subplots(1, 2)
        axs[0].imshow(self.current_interferogram, cmap='gray')
        axs[0].set_title('Simulated Interferogram (max_dev = %.2f nm)' % self.current_maximum_deviation_nm)
        axs[0].set_xlabel('Pixel')
        axs[0].set_ylabel('Pixel')
        axs[1].imshow(self.surface * 2 * np.pi * self.aperture_mask, cmap='jet')
        axs[1].set_title('Simulated Surface (in rads)')
        for mc in self.minima_curves:
            axs[0].plot(mc[:, 1], mc[:, 0], 'r--', linewidth=2)
        plt.show()
