import numpy as np
from scipy.ndimage import binary_fill_holes
from cv2 import GaussianBlur, normalize, NORM_MINMAX
import matplotlib.pyplot as plt
from Varios.optimizations import encontrar_maximo_cuadratica


def all_pixels_inside_border(border, shape):
    """
    Given a set of points that form the border of a closed shape,
    returns all the points (r, c) that are inside the shape.
    """
    border = np.round(np.array(border)).astype(int)
    rs_border = border[:, 0]
    cs_border = border[:, 1]

    img = np.zeros(shape, dtype=bool)
    img[rs_border, cs_border] = True

    filled = binary_fill_holes(img)

    rs_i, cs_i = np.where(filled)
    return np.column_stack([rs_i, cs_i])


def minimum_point_per_row(array_rc, values):
    """
    Given a set of points (r, c) and their corresponding values,
    returns the point with the minimum value for each unique row r.
    """
    min_row = np.min(array_rc[:, 0])
    max_row = np.max(array_rc[:, 0])
    rows = np.arange(min_row, max_row + 1)
    min_points = np.zeros((rows.size, 2), dtype=int)

    for index, r in enumerate(rows):
        mask = array_rc[:, 0] == r
        row_points = array_rc[mask]
        row_values = values[mask]

        min_index, _, _ = encontrar_maximo_cuadratica(row_points[:, 1], row_values, max_number_points=0, show=False,
                                                      extreme="min", extra_title=f"Row {r} in ({min_row}, {max_row})")
        min_points[index] = (r, min_index)

        # min_index = np.argmin(row_values)
        # min_points[index] = row_points[min_index]

    return min_points


def log_normalize(img, sigma=50):
    safe_img = img.astype(np.float32) + 1e-6
    logI = np.log(safe_img)
    logL = GaussianBlur(logI, (0, 0), sigmaX=sigma, sigmaY=sigma)
    logL = normalize(logL, None, norm_type=NORM_MINMAX) + 1e-6
    img = img / logL
    return img


def graphical_input_zones(img, help_text=None):
    """
    Allows the user to select rectangular areas on an image to define zones. The user is prompted to click twice on the
    image to select the top-left and bottom-right corners of the zones. Middle mouse button to end selection, right
    click to remove last point.

    :param img: Image on which to select zones
    :return: List of tuples of the form ((x1, y1), (x2, y2))
    """
    plt.imshow(img, cmap="gray")
    plt.title(
        "Select zones by clicking two points (top-left and bottom-right).\n"
        "Middle click to finish, right click to remove last point."
    )
    if help_text:
        plt.text(0.05, 0.95, help_text, transform=plt.gca().transAxes, fontsize=12, ha="left", va="top")
    points = []
    while True:
        selected = plt.ginput(2, timeout=0, show_clicks=True)
        if len(selected) < 2:
            break
        (x1, y1), (x2, y2) = selected
        x1, x2 = int(round(x1)), int(round(x2))
        y1, y2 = int(round(y1)), int(round(y2))
        points.append(((min(x1, x2), min(y1, y2)), (max(x1, x2), max(y1, y2))))
        plt.gca().add_patch(
            plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="red", linewidth=1.5,)
        )
        plt.draw()
    plt.close()
    return points
