#!/bin/bash

# Verificar si se pasaron suficientes argumentos
if [ "$#" -lt 6 ]; then
    echo "Uso: $0 <directorio> <x> <y> <ancho> <alto> <valor_maximo>"
    echo "Ejemplo: $0 /ruta/a/imagenes 50 50 250 250 255"
    exit 1
fi

# Argumentos
input_dir="$1"
x="$2"
y="$3"
width="$4"
height="$5"
max_value="$6"

# Verificar si el directorio existe
if [ ! -d "$input_dir" ]; then
    echo "Error: El directorio '$input_dir' no existe."
    exit 1
fi

# Crear subcarpeta 'cropped' dentro del directorio de entrada
output_dir="$input_dir/cropped"
mkdir -p "$output_dir"

# Procesar imágenes
for img in "$input_dir"/*.{jpg,jpeg,png,bmp,tiff,tif}; do
    [ -f "$img" ] || continue
    output_file="$output_dir/$(basename "$img")"
    min_value=$(convert "$img" -colorspace Gray -format "%[fx:minima]" info:)
    min_value_255=$(echo "$min_value * 255 / 1" | bc)
    convert "$img" -crop "${width}x${height}+${x}+${y}" -colorspace Gray -level ${min_value_255},${max_value},100% "$output_file"
    echo "Procesada: $img -> $output_file. Using min value = $min_value_255."
done

