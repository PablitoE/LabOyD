import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from mpl_toolkits.axes_grid1 import make_axes_locatable


# Full HD (1920x1080)
# Para HD (1280x720)
ROWS = 720
COLS = 1280
PHASE_AMPLITUDE = 1.5


def generate_linear_image(rows=ROWS, cols=COLS, reps_0_255=5):
    """
    Generar una imagen con resolucion HD donde los valores vayan de 0 a 255 linealmente 5 veces, con filas todas iguales
    """

    return np.tile(np.mod(np.linspace(0, 255 * reps_0_255, cols), 255), (rows, 1)).astype(np.uint8)


def generate_noise_image(rows=ROWS, cols=COLS, sigma_filter=200):
    """
    Generar una imagen con resolucion HD con valores aleatorios
    """
    img = np.random.randn(rows, cols)
    img = gaussian_filter(img, sigma=sigma_filter)
    img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255

    return img


def generate_block_image(rows=ROWS, cols=COLS, block_size=(120, 600)):
    """
    Generar una imagen con resolucion HD donde los valores vayan de 0 a 255 linealmente 5 veces, con filas todas iguales
    """
    img = np.zeros((rows, cols), dtype=np.uint8)
    img[
        rows // 2 - block_size[0] // 2: rows // 2 + block_size[0] // 2,
        cols // 2 - block_size[1] // 2: cols // 2 + block_size[1] // 2,
    ] = 255
    return img


def generate_carrier(fx, fy, rows=ROWS, cols=COLS):
    """
    Generar una imagen con resolucion HD donde los valores vayan linealmente en una dirección dada por dos frecuencias
    espaciales fx y fy
    """
    x = np.arange(cols)
    y = np.arange(rows)
    X, Y = np.meshgrid(x, y)
    carrier = 2 * np.pi * (fx * X + fy * Y)

    return carrier


if __name__ == "__main__":
    # img = generate_linear_image()
    # img = generate_noise_image()
    # fase_img = img.astype(np.float64) / 255 * PHASE_AMPLITUDE
    img = generate_block_image()
    fase_img = img.astype(np.float64) / 255 * PHASE_AMPLITUDE

    visibility = 0.7
    ruido = 0.04

    portadora = generate_carrier(0.015, 0.015)

    interferograma = 1 + visibility * np.cos(fase_img + portadora)

    interferograma = interferograma + ruido * np.random.randn(*interferograma.shape)
    interferograma = interferograma / np.max(interferograma) * 255


    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(img, cmap="gray", vmin=0, vmax=255)

    im = axs[1].imshow(interferograma, cmap="gray", vmin=0, vmax=255)
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)

    plt.tight_layout()
    plt.show()
