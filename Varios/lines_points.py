import numpy as np


def associate_two_sets_of_lines(lines1, lines2, flip=False):
    assert len(lines2) <= len(lines1), "El número de líneas en el segundo conjunto debe ser menor o igual que en el "\
                                       "primero."
    if flip:
        lines1 = [np.flip(line, axis=1) for line in lines1]
    # Calculate the rms_minimum distance between each pair of lines
    rmsmd_matrix = np.zeros((len(lines1), len(lines2)))
    distance_2_in_line1 = []
    for j, line2 in enumerate(lines2):
        for i, line1 in enumerate(lines1):
            # Calculate the matrix of distances between points on line1 and line2, being line1 and line2 arrays of
            # shape (n_points, 2)
            mat_dist = np.sqrt(
                (line1[:, 0].reshape(-1, 1) - line2[:, 0]) ** 2 + (line1[:, 1].reshape(-1, 1) - line2[:, 1]) ** 2
            )
            min_dist_2_in_line1 = np.min(mat_dist, axis=0)
            rmsmd_matrix[i, j] = np.sqrt(np.mean(min_dist_2_in_line1 ** 2))
            if i == 0:
                distance_2_in_line1.append(min_dist_2_in_line1)
                current_minimum_rmsmd = rmsmd_matrix[i, j]
            else:
                if rmsmd_matrix[i, j] < current_minimum_rmsmd:
                    distance_2_in_line1[j] = min_dist_2_in_line1
                    current_minimum_rmsmd = rmsmd_matrix[i, j]

    # For each line in lines2, find the line in lines1 with the minimum rmsmd
    min_rmsmd = np.argmin(rmsmd_matrix, axis=0)

    return min_rmsmd, distance_2_in_line1
