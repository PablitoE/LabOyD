import os
import sys
from glob import glob
from PIL import Image, ImageOps
import numpy as np


def process_images(input_dir, x, y, width, height, max_value):
    # Crear subcarpeta 'cropped' dentro del directorio de entrada
    output_dir = os.path.join(input_dir, "cropped")
    os.makedirs(output_dir, exist_ok=True)

    # Procesar imágenes soportadas
    image_formats = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif")
    images = []
    for fmt in image_formats:
        images.extend(glob(os.path.join(input_dir, fmt)))

    if not images:
        print(f"No se encontraron imágenes en {input_dir}.")
        return

    for img_path in images:
        try:
            # Abrir imagen
            with Image.open(img_path) as img:
                # Convertir a escala de grises
                gray_img = ImageOps.grayscale(img)

                # Recortar la imagen
                cropped = gray_img.crop((x, y, x + width, y + height))

                # Obtener valores mínimo y máximo
                np_img = np.array(cropped)
                min_value = np.min(np_img)

                # Obtener valor máximo como un porcentaje del valor promedio
                if max_value < 0:
                    max_value = -max_value
                    max_value = np.mean(np_img) * max_value

                # Escalar niveles de intensidad
                scale = 255 / (max_value - min_value)
                adjusted = np.clip((np_img - min_value) * scale, 0, 255).astype(np.uint8)
                adjusted_img = Image.fromarray(adjusted)

                # Guardar resultado
                output_file = os.path.join(output_dir, os.path.basename(img_path))
                adjusted_img.save(output_file)

                print(f"Procesada: {img_path} -> {output_file}.")

        except Exception as e:
            print(f"Error procesando {img_path}: {e}")


if __name__ == "__main__":
    # Verificar argumentos
    if len(sys.argv) < 7:
        print("Uso: python script.py <directorio> <x> <y> <ancho> <alto> <valor_maximo>")
        print("Ejemplo: python script.py /ruta/a/imagenes 50 50 250 250 255")
        print("El valor máximo puede ser negativo. Lo que significa que se calculará como"
              " una proporción del valor promedio (sin signo). Recomendado: -2.")
        sys.exit(1)

    # Leer argumentos
    input_dir = sys.argv[1]
    x = int(sys.argv[2])
    y = int(sys.argv[3])
    width = int(sys.argv[4])
    height = int(sys.argv[5])
    max_value = float(sys.argv[6])
    if max_value > 0:
        max_value = int(max_value)

    # Verificar si el directorio existe
    if not os.path.isdir(input_dir):
        print(f"Error: El directorio '{input_dir}' no existe.")
        sys.exit(1)

    # Procesar imágenes
    process_images(input_dir, x, y, width, height, max_value)
