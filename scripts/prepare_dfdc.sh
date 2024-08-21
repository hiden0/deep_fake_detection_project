#!/bin/bash

# Directorio principal donde se encuentran las carpetas
directorio_principal="/srv/hdd2/javber/dfdc/dfdc_train_all"

# Bucle para iterar sobre las carpetas
for carpeta in "$directorio_principal"/dfdc_train_part_* ; do
  # Verificamos si es un directorio
  if [ -d "$carpeta" ]; then
    # Obtenemos la ruta a la carpeta padre
    carpeta_padre=$(dirname "$carpeta")

    # Movemos los archivos MP4 a la carpeta padre
    find "$carpeta" -name "*.mp4" -exec mv {} "$carpeta_padre" \;
    
    echo "Los videos de la carpeta '$carpeta' se han movido correctamente a '$carpeta_padre'"
  fi
done