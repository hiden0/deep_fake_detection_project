#!/bin/bash

# Directorio principal donde se encuentran las subcarpetas con los archivos JSON
directorio_principal="/srv/hdd2/javber/dfdc/dfdc_train_all/"

# Archivo de salida donde se combinarán todos los JSON
archivo_salida="metadata_combinado.json"

# Crear el archivo de salida vacío
touch "$archivo_salida"

# Iterar sobre todas las subcarpetas y archivos JSON
for carpeta in "$directorio_principal"/*; do
  if [ -d "$carpeta" ]; then
    for archivo in "$carpeta"/*.json; do
      # Agregar el contenido de cada archivo JSON al final del archivo de salida
      cat "$archivo" >> "$archivo_salida"
    done
  fi
done