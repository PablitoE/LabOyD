import numpy as np


def associate_two_sets_of_lines(lines1, lines2):
    assert len(lines2) <= len(lines1), "El número de líneas en el segundo conjunto debe ser menor o igual que en el "\
                                       "primero."
    # Calculate the rms_minimum distance between each pair of lines
    rmsmd_matrix = np.zeros((len(lines1), len(lines2)))
    for i, line1 in enumerate(lines1):
        for j, line2 in enumerate(lines2):
            # Calculate the matrix of distances between points on line1 and line2, being line1 and line2 arrays of
            # shape (n_points, 2)
            mat_dist = np.sqrt(
                (line1[:, 0].reshape(-1, 1) - line2[:, 0]) ** 2 + (line1[:, 1].reshape(-1, 1) - line2[:, 1]) ** 2
            )
            min_dist_2_in_line1 = np.min(mat_dist, axis=0)
            rmsmd_matrix[i, j] = np.sqrt(np.mean(min_dist_2_in_line1 ** 2))
    # For each line in lines2, find the line in lines1 with the minimum rmsmd
    min_rmsmd = np.argmin(rmsmd_matrix, axis=0)

    return min_rmsmd
