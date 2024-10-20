#!/bin/bash

# Ruta base de los datos y el dataset
data_base="/srv/hdd2/javber/backup"
output_base="/srv/hdd2/javber/dataset/train_set"

# Bucle para iterar desde la parte 02 hasta la parte 35
for i in $(seq -f "%02g" 30 35)
do
    # Construir las rutas de entrada y salida para cada parte
    data_path="${data_base}/dfdc_train_part_${i}"
    output_path="${output_base}/dfdc_train_part_${i}"

    # Ejecutar el script de Python
    python /home/javber/repositorio/deep_fake_detection_project/Cross-Efficient-Vision-Transformer/preprocessing/extract_crops.py --data_path "$data_path" --output_path "$output_path"

    # Verificar si el comando anterior fue exitoso
    if [ $? -ne 0 ]; then
        echo "Error en la parte dfdc_train_part_${i}"
        exit 1
    else
        echo "Completado: dfdc_train_part_${i}"
    fi
done

echo "Proceso completado para todas las partes."
