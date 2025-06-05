import numpy as np
import matplotlib.pyplot as plt


def calcular_lente_divergente(apertura_numerica, diametro_colimador, f_colimador, f_divergente):
    # Calcular el diámetro objetivo del haz (10% mayor que el diámetro del colimador)
    diametro_salida = 1.1 * diametro_colimador
    distancia_pinhole_lente = -f_divergente * (diametro_salida - diametro_colimador) / diametro_salida
    radio_cono = distancia_pinhole_lente * apertura_numerica

    # Verificar que el cono no bloquee la lente colimadora y que el haz sea colimado
    angulo_divergencia = np.arctan(radio_cono / distancia_pinhole_lente)
    angulo_salida = np.arctan((diametro_salida / 2) / f_colimador)
    if 2 * radio_cono <= diametro_colimador and np.isclose(angulo_divergencia, angulo_salida, atol=0.001):
        return f_divergente, distancia_pinhole_lente, radio_cono, diametro_salida

    raise ValueError("El haz no es colimado o el cono de luz es más grande que el colimador.")


def graficar_cono(apertura_numerica, diametro_colimador, f_colimador, distancia_pinhole_lente, f_divergente, radio_cono, diametro_salida):
    # Crear el perfil del cono de luz antes y después de la lente divergente
    x_pinhole = 0
    x_lente = distancia_pinhole_lente
    x_colimador = distancia_pinhole_lente - f_divergente + f_colimador
    x_salida = x_colimador + 200  # Extender 200 mm después de la lente colimadora

    # Definir límites del gráfico
    plt.figure(figsize=(12, 6))
    plt.plot([x_pinhole, x_lente], [0, radio_cono], 'b-', linewidth=1.5)
    plt.plot([x_pinhole, x_lente], [0, -radio_cono], 'b-', linewidth=1.5)
    plt.plot([x_lente, x_colimador], [radio_cono, diametro_salida / 2], 'r-', linewidth=1.5)
    plt.plot([x_lente, x_colimador], [-radio_cono, -diametro_salida / 2], 'r-', linewidth=1.5)
    plt.plot([x_colimador, x_salida], [diametro_salida / 2, diametro_salida / 2], 'g--', linewidth=1.5)
    plt.plot([x_colimador, x_salida], [-diametro_salida / 2, -diametro_salida / 2], 'g--', linewidth=1.5)

    # Agregar etiquetas
    plt.axvline(x_lente, color='k', linestyle='--', label='Lente divergente')
    plt.axvline(x_colimador, color='g', linestyle='--', label='Lente colimadora')
    plt.axvline(x_pinhole, color='gray', linestyle='--', label='Pinhole')
    plt.title('Perfil del cono de luz desde el pinhole hasta después de la lente colimadora')
    plt.xlabel('Distancia (mm)')
    plt.ylabel('Radio del cono de luz (mm)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


def main():
    apertura_numerica = np.sin(np.atan(75e-3/2 * 1.1 / 250e-3))
    print(f"Apertura numérica: {apertura_numerica:.3f}")

    # Parámetros de la lente colimadora
    diametro_colimador = 100  # mm
    f_colimador = 200  # mm

    # Lista de lentes divergentes típicas (de -10 mm a -100 mm)
    distancias_focales = np.arange(-10, -105, -10)

    for f_divergente in distancias_focales:
        try:
            f_div, distancia_pinhole_lente, radio_cono, diametro_salida = calcular_lente_divergente(apertura_numerica, diametro_colimador, f_colimador, f_divergente)
            print(f"\nDistancia focal de la lente divergente: {f_div:.2f} mm")
            print(f"Posición de la lente divergente desde el pinhole: {distancia_pinhole_lente:.2f} mm")
            print(f"Diámetro del haz en el plano de salida de la lente colimadora: {diametro_salida:.2f} mm")
            graficar_cono(apertura_numerica, diametro_colimador, f_colimador, distancia_pinhole_lente, f_div, radio_cono, diametro_salida)            
        except ValueError as e:
            continue


if __name__ == "__main__":
    main()
