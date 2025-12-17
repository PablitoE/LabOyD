import FlechaInterfranja.flecha_interfranja as fei
import numpy as np
import matplotlib.pyplot as plt


def test_find_equidistant_peaks():
    # Test case 1: Equidistant peaks
    add_at = 5
    remove_at = 2
    data = np.random.rand(100)
    true_peaks_locs = np.arange(10, 90, 10)
    true_peaks_values = np.linspace(1.5, 3, len(true_peaks_locs))
    true_peaks_locs = np.delete(true_peaks_locs, remove_at)
    true_peaks_values = np.delete(true_peaks_values, remove_at)
    true_peaks_locs = np.insert(true_peaks_locs, add_at, true_peaks_locs[add_at] - 2)
    true_peaks_values = np.insert(true_peaks_values, add_at, true_peaks_values[add_at])
    data[true_peaks_locs] = true_peaks_values
    peaks = fei.find_equidistant_peaks(data, max_min='max', distance=5, prominence=0.7)
    plt.plot(data)
    plt.plot(true_peaks_locs, true_peaks_values, 'rx')
    plt.plot(peaks, data[peaks], 'go')
    plt.show()


if __name__ == '__main__':
    np.random.seed(1)
    test_find_equidistant_peaks()
