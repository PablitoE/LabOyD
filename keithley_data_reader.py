import pandas as pd
import matplotlib.pyplot as plt


class DataReader:
    # Definir colspecs como una constante de la clase
    COLSPECS = [(0, 7), (14, 21), (28, 35), (43, 50), (57, 64), (70, 79), (84, None)]

    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    def read_data(self):
        # Leer el archivo de texto usando colspecs
        self.df = pd.read_fwf(self.file_path, colspecs=self.COLSPECS, header=None)

        # Asignar nombres a las columnas
        self.df.columns = ['Col1', 'Col2', 'Col3', 'Col4', 'Col5', 'Hora', 'Fecha']

        # Convertir las columnas numéricas a float (las primeras 5)
        self.df[['Col1', 'Col2', 'Col3', 'Col4', 'Col5']] = (
            self.df[['Col1', 'Col2', 'Col3', 'Col4', 'Col5']]
            .replace({',': '.'}, regex=True)
            .apply(pd.to_numeric)
        )
        self.df[['Col1', 'Col2', 'Col3', 'Col4', 'Col5']] = self.df[
            ['Col1', 'Col2', 'Col3', 'Col4', 'Col5']
        ].apply(pd.to_numeric, errors='coerce')

        # Convertir la columna de hora en formato HH:MM:SS a tipo datetime
        self.df['Hora'] = pd.to_datetime(self.df['Hora'], format='%H:%M:%S').dt.time

        # Convertir la columna de fecha en formato DD:MM:AA a tipo datetime
        self.df['Fecha'] = pd.to_datetime(self.df['Fecha'], format='%d/%m/%y')

        self.df['FechaHora'] = pd.to_datetime(
            self.df['Fecha'].astype(str) + ' ' + self.df['Hora'].astype(str)
        )

    def get_data(self):
        # Retorna el DataFrame leído
        return self.df

    def plot_data(self, column='Col2'):
        plt.figure(figsize=(10, 6))
        plt.scatter(
            self.df['FechaHora'], self.df[column],
            color='#1E58E8', label=f'{column} vs FechaHora'
        )
        plt.plot(self.df['FechaHora'], self.df[column], color='#706CE8')
        plt.xlabel('Fecha y Hora')
        plt.ylabel(column)
        plt.title(f'Gráfico de {column} vs Fecha y Hora')
        plt.xticks(rotation=45)  # Rotar las etiquetas del eje x para que se vean mejor
        plt.legend()
        plt.tight_layout()
        plt.show()


# Ejemplo de uso
if __name__ == '__main__':
    # Instanciar la clase
    reader = DataReader(
        'Temperatura_salas_41_46_30_Nov24.dat'
    )

    # Leer los datos
    reader.read_data()

    reader.plot_data('Col3')
