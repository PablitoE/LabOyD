import numpy as np
from Varios.lines_points import associate_two_sets_of_lines


def test_associate_two_sets_of_lines():
    np.random.seed(0)
    n_sets_1 = 6
    min_n_points_1 = 30
    max_n_points_1 = 50
    lines_set_1 = []
    points_sets_1 = []

    for _ in range(n_sets_1):
        n_points = np.random.randint(min_n_points_1, max_n_points_1 + 1)
        points = np.random.rand(n_points, 2)
        line = np.polyfit(points[:, 0], points[:, 1], 1)
        new_points = points.copy()
        new_points[:, 1] = points[:, 0] * line[0] + line[1]
        lines_set_1.append(line)
        points_sets_1.append(new_points)

    n_sets_2 = 4
    assert n_sets_2 <= n_sets_1
    min_n_points_2 = 10
    max_n_points_2 = 20
    noise_level = 0.01
    true_lines_in_1 = np.random.choice(range(n_sets_1), n_sets_2, replace=False)
    points_sets_2 = []
    for line_idx in true_lines_in_1:
        n_points = np.random.randint(min_n_points_2, max_n_points_2 + 1)
        points = np.random.rand(n_points, 2)
        points[:, 1] = (
            points[:, 0] * lines_set_1[line_idx][0]
            + lines_set_1[line_idx][1]
            + np.random.normal(0, noise_level, size=n_points)
        )
        points_sets_2.append(points)

    min_rmsmd, distances = associate_two_sets_of_lines(points_sets_1, points_sets_2)
    assert np.array_equal(true_lines_in_1, min_rmsmd), f"Expected {true_lines_in_1}, but got {min_rmsmd}"
