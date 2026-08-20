from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

ARCHIVO_SALIDA = r'/home/pablo/OneDrive/Documentos/INTI-Varios/Presentaciones/Metrología Legal/superposicion_ondas.gif'

# 1. Definición de dominios y parámetros
# 5 periodos significan que x va de 0 a 5 * 2*pi = 10*pi
x = np.linspace(0, 10 * np.pi, 200)
t = x / 2 / np.pi

# Número de frames para la animación (fase de 0 a 2*pi)
duracion = 3
frame_rate = 25
n_frames = int(duracion * frame_rate)
phases = np.linspace(0, 2 * np.pi, n_frames)

# Curva 1: Fija (cos(x))
y1 = np.cos(x)

# Curva de amplitud en el segundo gráfico:
# La amplitud máxima de la suma de dos cosenoides con la misma amplitud es 2 * |cos(phi / 2)|
intensity_curve = 2 * np.cos(phases / 2)**2  # 2 + 2 * np.cos(phases)

# 2. Configuración de la figura y los subplots (dos columnas)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# --- Configuración del Plot Izquierdo (Tres curvas cosenoidales) ---
ax1.set_xlim(t[0], t[-1])
ax1.set_ylim(-2.2, 2.2)
ax1.set_title("Superposición de Ondas Cosenoidales (5 Periodos)", fontsize=12)
ax1.set_xlabel("t / periodos")
ax1.set_ylabel("Amplitud")
ax1.grid(True, linestyle='--', alpha=0.6)

# Inicializar líneas para las tres curvas
line1, = ax1.plot(t, y1, label=r'1ra: $\cos(x)$ (Fija)', color='blue', alpha=0.7)
line2, = ax1.plot(t, np.cos(x + phases[0]), label=r'2da: $\cos(x + \phi)$ (Variable)', color='orange', alpha=0.7)
line3, = ax1.plot(t, y1 + np.cos(x + phases[0]), label='3ra: Suma', color='green', linewidth=2)

# --- Configuración del Plot Derecho (Evolución de la amplitud y punto móvil) ---
ax2.set_xlim(0, 2 * np.pi)
ax2.set_ylim(-0.2, 2.2)
ax2.set_title(r"Intensidad media de la Tercera Curva, $I(\phi)$", fontsize=12)
ax2.set_xlabel(r'Fase inicial de la segunda curva, $\phi$ (radianes)')
ax2.set_ylabel('Intensidad media')
ax2.grid(True, linestyle='--', alpha=0.6)

# Graficar la curva fija de la amplitud a lo largo de todo el rango de fases
ax2.plot(phases, intensity_curve, color='purple', linestyle='-', label='Posibles intensidades medias')

# Puntos grandes que se va moviendo
point_phi0, = ax1.plot([], [], marker='o', markersize=10, color='red', label=r'Amplitud Inicial, $\cos(\phi)$')
point, = ax2.plot([], [], marker='o', markersize=10, color='red', label='Estado Actual')
ax1.legend(loc='upper right')
ax2.legend(loc='upper right')

# 3. Función de actualización para la animación
def update(frame):
    phi = phases[frame]

    # Actualizar la segunda y tercera curva en el plot izquierdo
    y2_curr = np.cos(x + phi)
    y3_curr = y1 + y2_curr

    line2.set_ydata(y2_curr)
    line3.set_ydata(y3_curr)

    # Actualizar la posición del punto móvil en el plot derecho
    current_intensity = np.mean(y3_curr ** 2)
    point.set_data([phi], [current_intensity])
    point_phi0.set_data([0], [y2_curr[0]])

    return line1, line2, line3, point, point_phi0

# 4. Creación de la animación
ani = animation.FuncAnimation(
    fig, update, frames=n_frames, interval=1000 / frame_rate, blit=True
)

# --- AGREGA ESTA LÍNEA PARA GUARDAR ---
ani.save(ARCHIVO_SALIDA, writer='pillow', fps=frame_rate)

path_primer_frame = Path(ARCHIVO_SALIDA).parent / 'primer_frame.png'
update(0)
plt.savefig(path_primer_frame, dpi=300, bbox_inches='tight')

plt.tight_layout()
plt.show()
