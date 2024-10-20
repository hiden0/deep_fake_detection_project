import numpy as np
import os
import pandas as pd
import shutil


def eliminar_carpeta_por_nombre_columna(df, directorio):
    """
    Itera sobre las columnas de un DataFrame y elimina una carpeta con el mismo nombre
    en el directorio especificado.

    Args:
        df (pd.DataFrame): El DataFrame a analizar.
        directorio (str): La ruta al directorio donde se buscarán y eliminarán las carpetas.
    """
    removed = 0
    for index, row in df.iterrows():
        # if columna == "name":
        name = row["name"][:-4]
        label = row["label"]
        ruta_carpeta = os.path.join(directorio, name)
        if name == "zrxngesosr":
            print("e")
        if label == 0:
            try:
                # os.rmdir(ruta_carpeta)
                shutil.rmtree(ruta_carpeta)
                print(f"Carpeta '{ruta_carpeta}' eliminada exitosamente.")
                removed += 1
            except OSError as error:
                # print(f"Error al eliminar la carpeta '{ruta_carpeta}': {error}")
                pass
    print(f"End of the process, removed {removed} videos")


# Ejemplo de uso:


DATA_DIR = "/srv/hdd2/javber/dataset/"
TRAINING_DIR = DATA_DIR + "train_set"
VALIDATION_LABELS_PATH = os.path.join(TRAINING_DIR, "metadata.csv")


df = pd.read_csv(VALIDATION_LABELS_PATH)
df = df.rename(columns={df.columns[0]: "name"})
directorio = "/srv/hdd2/javber/dataset/validation_set/dfdc_train_part_47/"  # Ajusta la ruta al directorio deseado

eliminar_carpeta_por_nombre_columna(df, directorio)
