import pickle
import numpy as np
import matplotlib.pyplot as plt
from FlechaInterfranja.flecha_interfranja import search_points_in_valley, analyze_interference


debug_fringe_search = False
debug_circle_detection = False
debug_failed_rotation = False
debug_failed_interferogram = True

# Debugging fringe search
if debug_fringe_search:
    with open("debug_interrupted_fringe_search.pkl", "rb") as f:
        data = pickle.load(f)

    data["aperture"] = 7
    data["step"] = 25
    output = search_points_in_valley(**data)

    plt.imshow(data['img'], cmap='gray')
    plt.plot(data['x'], data['y'], 'ro')
    for point in output:
        plt.plot(point[0], point[1], 'go')
    plt.show()

# Debugging circle detection
if debug_circle_detection:
    with open("2025-12-10_09-25-27_debug_no_circles_found.pkl", "rb") as f:
        data = pickle.load(f)

    import cv2

    blurred = data["image"]

    print(f"Todo cero? {np.all(blurred == 0)}")
    plt.imshow(blurred, cmap='gray')
    plt.show()

    circles = cv2.HoughCircles(
        blurred,
        data["method"],
        data["dp"],
        data["minDist"],
        param1=data["param1"],
        param2=data["param2"],
        minRadius=data["minRadius"],
        maxRadius=data["maxRadius"]
    )

    output = np.copy(blurred)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(output, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(output, (i[0], i[1]), 2, (0, 0, 255), 3)

        plt.imshow(output, cmap='gray')
        plt.show()
    else:
        raise ValueError("No se detectó ningún círculo en la imagen.")

# Debugging failed rotation
if debug_failed_rotation:
    with open("2025-12-10_09-46-47_debug_failed_rotation.pkl", "rb") as f:
        data = pickle.load(f)

    img = data["img"]
    ignore_low_freq_pixels = data["ignore_low_freq_pixels"]

    from FlechaInterfranja.flecha_interfranja import rotate_image_to_max_frequency

    img_rotada, angle_rotated = rotate_image_to_max_frequency(
        img,
        ignore_low_freq_pixels=ignore_low_freq_pixels,
        range_angle_deg=7,
        n_range_angle=15
    )

    print(f"Todo cero? {np.all(img_rotada == 0)}")
    plt.imshow(img_rotada, cmap='gray')
    plt.show()

# Debugging failed interferogram analysis
filename = "debug_failed_arrow.pkl"
if debug_failed_interferogram:
    with open(filename, "rb") as f:
        cnt_ims = 0
        while True:
            try:
                data = pickle.load(f)
                cnt_ims += 1
            except EOFError:
                break

    with open(filename, "rb") as f:
        cnt = 1
        while True:
            try:
                data = pickle.load(f)
            except EOFError:
                break

            print(f"Imagen {cnt} de {cnt_ims}")
            interferogram = data["interferogram"]
            debugging_info = data["debugging_info"]
            # from scipy.ndimage import rotate
            # angle_deg_rotate = 10
            # interferogram = rotate(interferogram, angle_deg_rotate, mode='nearest', reshape=False)
            print(
                f"Arrow (px): {debugging_info['arrow']}, Simulated arrow (px): {debugging_info['simulated_arrow_px']}"
            )
            N_REGULARIZERS = 20
            arrows = np.zeros(N_REGULARIZERS)
            interfringes = np.zeros(N_REGULARIZERS)
            regularizers = np.logspace(-2, 1, N_REGULARIZERS)
            for k_reg, reg_param in enumerate(regularizers):
                interfringe, arrow = analyze_interference(
                    image_array=interferogram, save=False, show_result=False, show=False, debugging_info=debugging_info,
                    regularizer_parameter=reg_param,
                )
                interfringes[k_reg] = interfringe.n  # ufloat to float
                arrows[k_reg] = arrow.n  # ufloat to float

            fig, axs = plt.subplots(1, 2, figsize=(12, 5))
            axs[0].semilogx(regularizers, interfringes, marker='o')
            axs[0].set_xlabel("Regularizer parameter")
            axs[0].set_ylabel("Interfringe distance (px)")
            axs[0].set_title("Interfringe distance vs Regularizer parameter")
            axs[0].axhline(
                debugging_info["simulated_interfringe_spacing"], color='red', linestyle='--',
                label='Simulated interfringe',
            )
            axs[0].legend()
            axs[1].semilogx(regularizers, arrows, marker='o', color='orange')
            axs[1].set_xlabel("Regularizer parameter")
            axs[1].set_ylabel("Arrow (px)")
            axs[1].set_title("Arrow vs Regularizer parameter")
            axs[1].axhline(debugging_info["simulated_arrow_px"], color='red', linestyle='--', label='Simulated arrow')
            axs[1].legend()
            plt.suptitle(f"Debugging interferogram {cnt}")
            plt.show()

            cnt += 1
