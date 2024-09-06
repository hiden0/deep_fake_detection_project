#!/bin/bash

# Ruta base
base_path="/srv/nvme/javber/dfdc_train_part_"

# Bucle sobre 49 carpetas
for i in $(seq -w 43 44); do
    # Construir la ruta de la carpeta actual
    folder="${base_path}${i}"
    
    # Ejecutar el comando para la carpeta actual
    python /home/javber/repositorio/deep_fake_detection_project/Cross-Efficient-Vision-Transformer/preprocessing/detect_faces_facenet.py --data_path "$folder"
done