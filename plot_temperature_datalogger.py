import pandas as pd
import matplotlib.pyplot as plt

# Leer el archivo CSV
archivo_path = r"Data/DataLogger_Dimensional/DIM-01-MUL_1000_21_11_24.csv"

df = pd.read_csv(archivo_path, skiprows=1)

# Convertir la columna de fechas a tipo datetime (si no está ya en ese formato)
df['Fecha'] = pd.to_datetime(df['Fecha'])

# Crear el gráfico de dispersión
plt.figure(figsize=(10, 6))
plt.scatter(df['Fecha'], df['Temperatura'], color='#A7C8D9', label='Temperatura',
            alpha=0.7, s=20)
plt.plot(df['Fecha'], df['Temperatura'], color='#5A9BD5', alpha=0.7, linestyle='--')
plt.title('Gráfico de Dispersión de Temperaturas')
plt.xlabel('Fecha')
plt.ylabel('Temperatura (°C)')
plt.xticks(rotation=45)  # Rotar las fechas para mejor visualización
plt.grid(True)
plt.legend()
plt.tight_layout()  # Ajustar los márgenes
plt.show()
