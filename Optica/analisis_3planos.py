"""
Se genera una superficie y se analiza si se resuelve bien la desviación máxima de planitud usando el método de los 3
planos cuando se usan modelos diferentes.
"""
import logging

import matplotlib.pyplot as plt
import numpy as np
import tqdm
from scipy.ndimage import gaussian_filter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_surface(n_rows, n_cols, max_dev_pair, max_dev_impair):
    x = np.linspace(-1.1, 1.1, n_cols)
    y = np.linspace(-1.1, 1.1, n_rows)
    xx, yy = np.meshgrid(x, y)
    surface = np.random.randn(n_rows, n_cols)
    surface = gaussian_filter(surface, sigma=np.sqrt(n_rows * n_cols) / 8)
    """
    Tomar la parte par de la superficie como la que reflejada en x da igual, y la parte impar de la superficie como la
    que reflejada en y da 0.
    """
    surface_pair = (surface + surface[:, ::-1]) / 2
    surface_impair = surface - surface_pair
    surface_pair = surface_pair / (np.max(surface_pair) - np.min(surface_pair))
    surface_impair = surface_impair / (np.max(surface_impair) - np.min(surface_impair))
    surface_pair *= max_dev_pair
    surface_impair *= max_dev_impair
    return surface_pair + surface_impair

if __name__ == "__main__":
    verbose = False
    n_rows = 256
    n_cols = 256
    max_dev_pair_k = 54
    max_dev_impair_k = 0
    max_dev_pair_l = 68
    max_dev_impair_l = 0
    max_dev_pair_m = 57
    max_dev_impair_m = 0
    uncert_k2 = 0

    n_tests = 1000
    error_m = np.zeros(n_tests)
    error_l = np.zeros(n_tests)
    error_k = np.zeros(n_tests)

    for kt in tqdm.tqdm(range(n_tests)):
        surface_k = generate_surface(n_rows, n_cols, max_dev_pair_k, max_dev_impair_k)
        surface_l = generate_surface(n_rows, n_cols, max_dev_pair_l, max_dev_impair_l)
        surface_m = generate_surface(n_rows, n_cols, max_dev_pair_m, max_dev_impair_m)

        max_dev_k = np.ptp(surface_k)
        max_dev_l = np.ptp(surface_l)
        max_dev_m = np.ptp(surface_m)

        max_dev_D = np.abs(np.max(np.abs(surface_k + surface_l[:, ::-1])) + np.random.randn() * uncert_k2 / 2)
        max_dev_E = np.abs(np.max(np.abs(surface_m + surface_l[:, ::-1])) + np.random.randn() * uncert_k2 / 2)
        max_dev_G = np.abs(np.max(np.abs(surface_m + surface_k[:, ::-1])) + np.random.randn() * uncert_k2 / 2)
        if verbose:
            logger.info("Nomenclatura de Fritz (1984):")
            logger.info(f"Measured D (k + l-): {max_dev_D}")
            logger.info(f"Measured E (m + l-): {max_dev_E}")
            logger.info(f"Measured G (m + k-): {max_dev_G}")

        k_measured = (max_dev_D - max_dev_E + max_dev_G) / 2
        l_measured = (max_dev_D + max_dev_E - max_dev_G) / 2
        m_measured = (- max_dev_D + max_dev_E + max_dev_G) / 2

        if verbose:
            logger.info(f"Measured k: {k_measured}")
            logger.info(f"Measured l: {l_measured}")
            logger.info(f"Measured m: {m_measured}")
        error_k[kt] = k_measured - max_dev_k
        error_l[kt] = l_measured - max_dev_l
        error_m[kt] = m_measured - max_dev_m

    error_k = np.sqrt(np.mean(error_k**2))
    error_l = np.sqrt(np.mean(error_l**2))
    error_m = np.sqrt(np.mean(error_m**2))
    logger.info(f"Error k: {error_k}")
    logger.info(f"Error l: {error_l}")
    logger.info(f"Error m: {error_m}")
