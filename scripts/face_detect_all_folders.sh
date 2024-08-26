#!/bin/bash

# Ruta base
base_path="/data/dfdc_train_part_"

# Bucle sobre 49 carpetas
for i in $(seq -w 07 09); do
    # Construir la ruta de la carpeta actual
    folder="${base_path}${i}"
    
    # Ejecutar el comando para la carpeta actual
    python /app/Cross-Efficient-Vision-Transformer/preprocessing/detect_faces_mediapipe.py --data_path "$folder"
done