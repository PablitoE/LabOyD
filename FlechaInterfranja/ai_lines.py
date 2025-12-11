from Varios import ai_based_optimization as abo
from FlechaInterfranja.interferogram_generation import FlatInterferogramGenerator
import torch
import logging
import os


if __name__ == "__main__":
    SAVE_PATH = "Data/Resultados/AI_FEI"
    os.makedirs(SAVE_PATH, exist_ok=True)
    EPOCHS = 1000        
    N_IMS_PER_SAMPLE = 10
    MIN_N_FRINGES = 7
    MAX_N_FRINGES = 20
    MAX_ROTATION_DEG = 1
    VISIBILITY_RATIO = 0.5
    NOISE_LEVEL = 0.001
    WAVELENGTH_NM = 632.8
    MAX_DEVIATION_NM = 150.0
    PLOT_INTERFEROGRAMS = False
    PLOT_SURFACES = False
    PLOT_FITS = False
    PLOT_FRINGE_DETECTION = False
    PLOT_RESULTS = True
    SIMULATION_MODE = "random"  # "random", "random_maxfixed", "gaussian"

    logging.basicConfig(
        level=logging.CRITICAL,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    device = torch.device("cpu")
    model = abo.CurveSetModel().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()    

    generator = FlatInterferogramGenerator(shape=(1024, 1024),
                                                   pixel_size=65e-6,
                                                   save_path=SAVE_PATH,
                                                   wavelength_nm=632.8,
                                                   min_fringe=MIN_N_FRINGES,
                                                   max_fringe=MAX_N_FRINGES,
                                                   diameter=50e-3,
                                                   max_rotation=MAX_ROTATION_DEG,
                                                   visibility_ratio=VISIBILITY_RATIO,
                                                   noise_level=NOISE_LEVEL,
                                                   maximum_deviation_nm=MAX_DEVIATION_NM
                                                   )
    

    for epoch in range(EPOCHS):
        generator.prepare_next_generation(mode=SIMULATION_MODE)
        for k_interferogram, interferogram in enumerate(generator.generate(
            num_samples=N_IMS_PER_SAMPLE, output_mode="array", simulation_mode=SIMULATION_MODE,
            surface_options={"plot_surface": PLOT_SURFACES}
        )):
            batch_curves = generator.minima_curves
            #TODO: Sacar la componente constante y lineal de la superficie simulada
            opt.zero_grad()
            m_pred, b0_pred, d_pred = model(batch_curves)
            loss = (m_pred - m_true)**2 + (b0_pred - b0_true)**2 + (d_pred - delta_true)**2
            loss.backward()
            opt.step()
