from Varios.interferogram_generation import FlatInterferogramGenerator


if __name__ == "__main__":
    import pickle

    with open("2026-02-10_11-24-04_debug_insufficient_valley_curves.pkl", "rb") as f:
        data = pickle.load(f)
    
    phase_map = data["phase_map"]