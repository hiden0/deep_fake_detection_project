import json
import os
import shutil


def dividir_videos_por_etiqueta(carpeta_principal, archivo_metadata):
    """
    Divide los videos (representados por carpetas de frames) en subcarpetas según su etiqueta en el metadata.json, considerando la estructura de subcarpetas principal y eliminando la extensión .mp4 de los nombres de las carpetas.

    Args:
      carpeta_principal: Ruta a la carpeta principal que contiene las subcarpetas de videos.
      archivo_metadata: Ruta al archivo metadata.json.
    """

    # Cargar el archivo metadata.json
    with open(archivo_metadata, "r") as f:
        metadata = json.load(f)

    # Obtener las subcarpetas principales
    subcarpetas_principales = ["training_set"]

    # Iterar sobre las subcarpetas principales
    for subcarpeta_principal in subcarpetas_principales:
        ruta_subcarpeta_principal = os.path.join(
            carpeta_principal, subcarpeta_principal
        )

        # Crear las subcarpetas REAL y FAKE dentro de cada subcarpeta principal
        os.makedirs(os.path.join(ruta_subcarpeta_principal, "Original"), exist_ok=True)
        os.makedirs(os.path.join(ruta_subcarpeta_principal, "Deepfakes"), exist_ok=True)

        # Iterar sobre las anotaciones y mover las carpetas de frames
        for video in metadata:
            etiqueta = metadata[video]["label"]
            #         nombre_video = video
            nombre_video_sin_extension = video.split(".")[
                0
            ]  # Eliminar la extensión .mp4
            ruta_origen = os.path.join(
                ruta_subcarpeta_principal, nombre_video_sin_extension
            )
            ruta_destino = os.path.join(
                ruta_subcarpeta_principal, etiqueta, nombre_video_sin_extension
            )
            try:
                shutil.move(ruta_origen, ruta_destino)
            except:
                pass


# Ajusta las rutas según tu estructura de carpetas
carpeta_principal = "/srv/nvme/javber/deep_fake_detection_sample/dataset/test_set"
archivo_metadata = os.path.join(carpeta_principal, "metadata.json")

dividir_videos_por_etiqueta(carpeta_principal, archivo_metadata)


# import json
# import os
# import shutil


# def dividir_videos_por_etiqueta(carpeta_principal, archivo_metadata):
#     """
#     Divide los videos en subcarpetas según su etiqueta en el metadata.json.

#     Args:
#       carpeta_principal: Ruta a la carpeta principal que contiene los videos.
#       archivo_metadata: Ruta al archivo metadata.json.
#     """

#     # Cargar el archivo metadata.json
#     with open(archivo_metadata, "r") as f:
#         metadata = json.load(f)

#     # Crear las subcarpetas si no existen
#     os.makedirs(os.path.join(carpeta_principal, "REAL"), exist_ok=True)
#     os.makedirs(os.path.join(carpeta_principal, "FAKE"), exist_ok=True)

#     # Iterar sobre las anotaciones y mover los videos a la carpeta correspondiente
#     for video in metadata:
#         etiqueta = metadata[video]["label"]
#         nombre_video = video  # Ajusta esto al nombre del campo que contiene el nombre del video en tu JSON
#         ruta_origen = os.path.join(carpeta_principal, nombre_video)
#         ruta_destino = os.path.join(carpeta_principal, etiqueta)
#         shutil.move(ruta_origen, ruta_destino)


# # Ajusta las rutas según tu estructura de carpetas
# carpeta_principal = "/srv/nvme/javber/deep_fake_detection_sample/train_sample_videos/"
# archivo_metadata = os.path.join(carpeta_principal, "metadata.json")

# dividir_videos_por_etiqueta(carpeta_principal, archivo_metadata)
