import numpy as np

from FlechaInterfranja.interferogram_generation import FlatInterferogramGenerator

if __name__ == "__main__":
    import pickle

    with open("2026-02-11_12-22-40_debug_insufficient_valley_curves.pkl", "rb") as f:
        data = pickle.load(f)

    phase_map = data["phase_map"]

    n_fringes = 12
    phase_map_synthetic = np.linspace(0, 2 * np.pi * (n_fringes + 1), phase_map.shape[1], endpoint=False)
    phase_map_synthetic = phase_map_synthetic.reshape(1, phase_map.shape[1])
    phase_map_synthetic = np.tile(phase_map_synthetic, (phase_map.shape[0], 1))

    ig = FlatInterferogramGenerator(save_path='Data/Resultados/Debug')
    ig.current_phase = phase_map_synthetic

    ig.get_maximum_simulated_deviation_px(plot=True)
