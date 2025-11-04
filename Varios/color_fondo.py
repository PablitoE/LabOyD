from PIL import Image
import numpy as np
import os


def cambiar_fondo_proporcional(ruta_imagen, cp, cf, cd):
    """
    Cambia el fondo de una imagen de forma proporcional.

    Args:
        ruta_imagen (str): La ruta a la imagen de entrada.
        cp (tuple): El color principal en formato RGB (r, g, b).
        cf (tuple): El color de fondo actual en formato RGB (r, g, b).
        cd (tuple): El color de fondo deseado en formato RGB (r, g, b).
    """
    try:
        # Abrir la imagen
        img = Image.open(ruta_imagen).convert('RGB')
        # Convertir la imagen a un array de NumPy para facilitar los cálculos
        data = np.array(img, dtype=np.float64)

        # Convertir los colores a arrays de NumPy
        cp = np.array(cp, dtype=np.float64)
        cf = np.array(cf, dtype=np.float64)
        cd = np.array(cd, dtype=np.float64)

        # Calcular el vector de diferencia entre el fondo actual y el principal
        vector_cf_cp = cp - cf
        # Calcular el vector de diferencia entre el fondo deseado y el principal
        vector_cd_cp = cp - cd

        # Evitar la división por cero si el vector es nulo
        if np.all(vector_cf_cp == 0):
            # Si el color principal y el de fondo son iguales, no se puede calcular la proporción.
            # Se podría manejar este caso de diferentes maneras, aquí simplemente no se modifica la imagen.
            print("El color principal y el de fondo actual son idénticos, no se puede realizar la conversión.")
            return

        # Calcular la proporción de la distancia de cada píxel al color de fondo actual,
        # en relación con la distancia total entre el color principal y el de fondo actual.
        # Suponemos que cada píxel (p) se puede expresar como p = cf + t * (cp - cf)
        # Despejamos t = (p - cf) / (cp - cf)
        with np.errstate(divide='ignore', invalid='ignore'):
            # Calculamos (p - cf) para cada píxel
            diff_p_cf = data - cf
            # Calculamos la proporción 't'
            # Usamos np.divide para manejar la división por cero donde el componente de vector_cf_cp es 0
            # En esos casos, el ratio será 0 si el numerador también es 0, o infinito si no lo es.
            # np.nan_to_num convierte los NaN y los infinitos a números finitos.
            ratio = np.nan_to_num(np.divide(diff_p_cf, vector_cf_cp))

        # Limitamos el ratio al rango [0, 1] para evitar colores fuera de la gama esperada
        ratio = np.clip(ratio, 0, 1)

        # Aplicar la transformación al nuevo espacio de color
        # El nuevo color de cada píxel (p_nuevo) será: p_nuevo = cd + t * (cp - cd)
        nuevo_data = cd + ratio * vector_cd_cp

        # Asegurarse de que los valores de los píxeles estén en el rango válido de 0-255
        nuevo_data = np.clip(nuevo_data, 0, 255)

        # Convertir el array de nuevo a una imagen y al formato de 8 bits
        nueva_img = Image.fromarray(nuevo_data.astype(np.uint8), 'RGB')

        # Guardar la nueva imagen
        directorio, nombre_archivo = os.path.split(ruta_imagen)
        nombre_base, extension = os.path.splitext(nombre_archivo)
        nuevo_nombre_archivo = f"{nombre_base}_modificado{extension}"
        ruta_salida = os.path.join(directorio, nuevo_nombre_archivo)
        nueva_img.save(ruta_salida)
        print(f"Imagen guardada exitosamente en: {ruta_salida}")

    except FileNotFoundError:
        print(f"Error: No se encontró el archivo en la ruta: {ruta_imagen}")
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")


# --- Ejemplo de uso ---
if __name__ == '__main__':
    # Pide al usuario la ruta de la imagen
    # ruta_imagen_input = input("Introduce la ruta de la imagen: ")
    ruta_imagen_input = r'/home/pablo/OneDrive/Documentos/INTI-Varios/Internship SIM NIST/Work/sim logo.jpg'

    # Definición de los colores (puedes cambiarlos según tus necesidades)
    # cp: Color Principal (ej: el color del objeto que quieres conservar)
    # cf: Color de Fondo Actual (ej: el color del fondo que quieres cambiar)
    # cd: Color de Fondo Deseado (ej: el nuevo color para el fondo)

    # Ejemplo 1: Cambiar un fondo azul a uno verde en una imagen con un objeto rojo.
    color_principal = (0, 120, 0)   # Verde
    color_fondo_actual = (255, 255, 255)  # Blanco
    color_fondo_deseado = (33, 33, 33)  # Gris oscuro

    print("\nUsando colores de ejemplo:")
    print(f"Color Principal (cp): {color_principal}")
    print(f"Color de Fondo Actual (cf): {color_fondo_actual}")
    print(f"Color de Fondo Deseado (cd): {color_fondo_deseado}")

    # Llama a la función
    cambiar_fondo_proporcional(ruta_imagen_input, color_principal, color_fondo_actual, color_fondo_deseado)
