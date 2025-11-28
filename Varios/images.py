import numpy as np
from scipy.ndimage import binary_fill_holes


def all_pixels_inside_border(border, shape):
    """
    Given a set of points that form the border of a closed shape,
    returns all the points (x, y) that are inside the shape.
    """
    border = np.array(border)
    xs_border = border[:, 0]
    ys_border = border[:, 1]

    img = np.zeros(shape, dtype=bool)
    img[ys_border, xs_border] = True

    filled = binary_fill_holes(img)

    ys_i, xs_i = np.where(filled)
    return np.column_stack([xs_i, ys_i])
