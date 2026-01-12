import numpy as np
from scipy.ndimage import binary_fill_holes
from cv2 import GaussianBlur, normalize, NORM_MINMAX


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

        min_index = np.argmin(row_values)
        min_points[index] = row_points[min_index]

    return min_points


def log_normalize(img, sigma=50):
    safe_img = img.astype(np.float32) + 1e-6
    logI = np.log(safe_img)
    logL = GaussianBlur(logI, (0, 0), sigmaX=sigma, sigmaY=sigma)
    logL = normalize(logL, None, norm_type=NORM_MINMAX) + 1e-6
    img = img / logL
    return img
