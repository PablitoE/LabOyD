import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import MouseButton
from PIL import Image
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import rotate
from scipy.optimize import minimize


# Función para cargar una imagen a color en un array de NumPy
def load_color_image(image_path):
    return np.array(Image.open(image_path))

# Función para mostrar una figura con dos subfiguras donde la primera contiene una imagen a color y permite tomar dos
# puntos con ginput.
# En la segunda subfigura se muestran las componentes R, G y B de la imagen a color en el perfil de la linea que va de
# los puntos seleccionados.
# Cada vez que se selecciona un nuevo par de puntos se muestran las componentes R, G y B de la imagen a color en el
# perfil de la linea que va de los nuevos puntos seleccionados.
def profile_image(image: np.ndarray):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.imshow(image)
    title_ax1 = 'a color' if image.ndim == 3 else 'a escala de grises'
    ax1.set_title(f'Imagen {title_ax1}. Seleccione dos puntos.')
    ax2.set_title('Componentes R, G y B')

    x = np.arange(image.shape[1])
    y = np.arange(image.shape[0])
    if image.ndim == 3:
        interpolator_r = RectBivariateSpline(x, y, image[:, :, 0].T)
        interpolator_g = RectBivariateSpline(x, y, image[:, :, 1].T)
        interpolator_b = RectBivariateSpline(x, y, image[:, :, 2].T)
    else:
        interpolator_gray = RectBivariateSpline(x, y, image.T)
    a_line = None

    while True:
        points = plt.ginput(2, timeout=0, show_clicks=True, mouse_stop=MouseButton.RIGHT,
                            mouse_pop=MouseButton.MIDDLE)
        if not points or len(points) < 2:
            break

        good_points = np.array(points)
        if a_line:
            a_line[0].remove()
        a_line = ax1.plot(good_points[:, 0], good_points[:, 1], 'ro-')
        x1, y1 = good_points[0]
        x2, y2 = good_points[1]

        # Obtener el perfil en los tres colores haciendo una interpolación con paso de 1 pixel
        d = np.linalg.norm(np.array([x2 - x1, y2 - y1]))
        xs = np.linspace(x1, x2, int(d) + 1)
        ys = np.linspace(y1, y2, int(d) + 1)

        ax2.clear()
        if image.ndim == 3:
            r_profile = interpolator_r.ev(xs, ys)
            g_profile = interpolator_g.ev(xs, ys)
            b_profile = interpolator_b.ev(xs, ys)

            ax2.plot(r_profile, label='R', color='r')
            ax2.plot(g_profile, label='G', color='g')
            ax2.plot(b_profile, label='B', color='b')
            ax2.plot(0.299 * r_profile + 0.587 * g_profile + 0.114 * b_profile, label='Gris Default', color='k')
            ax2.plot(0.5 * r_profile + 0.5 * g_profile, label='R=0.5 G=0.5 B=0', color='y')
        else:
            gray_profile = interpolator_gray.ev(xs, ys)
            ax2.plot(gray_profile, label='Gris', color='k')
        ax2.legend()
        plt.draw()

    plt.close()
    if len(good_points) == 2:
        if image.ndim == 3:
            return good_points, (r_profile, g_profile, b_profile)
        else:
            return good_points, (gray_profile,)
    else:
        return None


# Función para hacer un ajuste de una curva con un polinomio cuadratico y obtener el residuo MSE
def ajustar_curva(x, y):
    p = np.polyfit(x, y, 2)
    y_ajustada = np.polyval(p, x)
    mse = np.mean((y - y_ajustada) ** 2)
    return p, mse


def mse_quad_linear_combination(rg_coefs, r_profile, g_profile, b_profile):
    intensity = rg_coefs[0] * r_profile + rg_coefs[1] * g_profile + (1 - rg_coefs[0] - rg_coefs[1]) * b_profile
    x = np.arange(len(r_profile))
    p, mse = ajustar_curva(x, intensity)
    return mse


# Función para encontrar la combinación lineal de las componentes R, G y B que minimice el residuo MSE. La suma de los
# coeficientes que multiplican a R, G y B es 1.
def find_linear_combination(r_profile, g_profile, b_profile):
    bounds = ((0, 1), (0, 1))
    constraints = {'type': 'ineq', 'fun': lambda rg_coefs: 1 - np.sum(rg_coefs)}
    c0 = np.array([0.299, 0.587])
    costos = []

    def register_mse(rg_coefs):
        intensity = rg_coefs[0] * r_profile + rg_coefs[1] * g_profile + (1 - rg_coefs[0] - rg_coefs[1]) * b_profile
        x = np.arange(len(r_profile))
        p, mse = ajustar_curva(x, intensity)
        costos.append(mse)

    res = minimize(mse_quad_linear_combination, c0, args=(r_profile, g_profile, b_profile), bounds=bounds,
                   constraints=constraints, callback=register_mse)
    return np.append(res.x, 1 - np.sum(res.x)), np.array(costos)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=str, help='Path to the image file')
    parser.add_argument('--angle', type=float, default=0, help='Rotation angle in degrees')
    args = parser.parse_args()

    image = load_color_image(args.image_path)
    if image.ndim == 3:
        image_custom = image.astype(np.float32)
        image_custom = image_custom[..., 0] * 0.5 + image_custom[..., 1] * 0.5
        plt.imshow(image_custom, cmap='gray')
        plt.show()

    if args.angle != 0:
        rotated_img = rotate(image, args.angle, mode='nearest', reshape=False)
    else:
        rotated_img = image
    points, perfiles = profile_image(rotated_img)
    if image.ndim == 3:
        coefs, evol_mse = find_linear_combination(perfiles[0], perfiles[1], perfiles[2])
        intensity = coefs[0] * perfiles[0] + coefs[1] * perfiles[1] + coefs[2] * perfiles[2]

        fig, axs = plt.subplots(1, 3, figsize=(15, 7))
        axs[0].imshow(rotated_img)
        axs[0].set_title('Imagen a color')
        axs[0].plot(points[:, 0], points[:, 1], 'ro-', label='Puntos seleccionados')
        axs[0].legend()
        coefs_str = 'R={:.3f}, G={:.3f}, B={:.3f}'.format(coefs[0], coefs[1], coefs[2])
        axs[1].set_title('Perfiles. Coeficientes : {}'.format(coefs_str))
        axs[1].plot(perfiles[0], label='R', linestyle=':')
        axs[1].plot(perfiles[1], label='G', linestyle=':')
        axs[1].plot(perfiles[2], label='B', linestyle=':')
        axs[1].plot(intensity, label='Intensidad ajustada')
        axs[1].legend()
        axs[2].plot(evol_mse)
        axs[2].set_title('Evolucion del MSE')
        plt.show()


if __name__ == '__main__':
    main()
