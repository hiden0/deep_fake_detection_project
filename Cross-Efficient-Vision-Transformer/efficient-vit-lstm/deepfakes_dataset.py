import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import cv2
import numpy as np
import pandas as pd
import os
import re
import uuid
from albumentations import (
    Compose,
    RandomBrightnessContrast,
    HorizontalFlip,
    FancyPCA,
    HueSaturationValue,
    OneOf,
    ToGray,
    ShiftScaleRotate,
    ImageCompression,
    PadIfNeeded,
    GaussNoise,
    GaussianBlur,
    Rotate,
)
from typing import Tuple
from transforms.albu import IsotropicResize
from PIL import Image
from torchvision import transforms


class DeepFakesDataset(Dataset):
    def __init__(
        self, data_root, labels_csv, image_size, sequence_length, mode="train"
    ):
        self.data_root = data_root
        self.labels = pd.read_csv(labels_csv)
        self.image_size = image_size
        self.sequence_length = sequence_length
        self.mode = mode

        # DESCOMENTAR SI TODOS LOS VIDEOS SE ENCUENTRAN EN UNA SOLA SUBCARPETA
        # self.video_dirs = [
        #     os.path.join(data_root, d)
        #     for d in os.listdir(data_root)
        #     if os.path.isdir(os.path.join(data_root, d))
        # ]

        # OPCION PARA CUANDO LOS VIDEOS SE ENCUENTRAN EN CARPETAS DIVIDIDAS EN 50 FRACCIONES
        self.video_dirs = []
        for subcarpeta in os.listdir(data_root):
            ruta_subcarpeta = os.path.join(data_root, subcarpeta)
            if os.path.isdir(ruta_subcarpeta):
                for subsubcarpeta in os.listdir(ruta_subcarpeta):
                    ruta_subsubcarpeta = os.path.join(ruta_subcarpeta, subsubcarpeta)
                    if os.path.isdir(ruta_subsubcarpeta):
                        self.video_dirs.append(ruta_subsubcarpeta)

    def create_train_transforms(self, size):
        additional_targets = {
            f"image_{i+1}": f"image_{i+1}" for i in range(self.sequence_length - 1)
        }
        return Compose(
            [
                ImageCompression(quality_lower=60, quality_upper=100, p=0.2),
                GaussNoise(p=0.3),
                # GaussianBlur(blur_limit=3, p=0.05),
                HorizontalFlip(),
                OneOf(
                    [
                        IsotropicResize(
                            max_side=size,
                            interpolation_down=cv2.INTER_AREA,
                            interpolation_up=cv2.INTER_CUBIC,
                        ),
                        IsotropicResize(
                            max_side=size,
                            interpolation_down=cv2.INTER_AREA,
                            interpolation_up=cv2.INTER_LINEAR,
                        ),
                        IsotropicResize(
                            max_side=size,
                            interpolation_down=cv2.INTER_LINEAR,
                            interpolation_up=cv2.INTER_LINEAR,
                        ),
                    ],
                    p=1,
                ),
                PadIfNeeded(
                    min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT
                ),
                OneOf(
                    [RandomBrightnessContrast(), FancyPCA(), HueSaturationValue()],
                    p=0.4,
                ),
                ToGray(p=0.2),
                ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.2,
                    rotate_limit=5,
                    border_mode=cv2.BORDER_CONSTANT,
                    p=0.5,
                ),
            ],
            additional_targets=additional_targets,
        )

    def create_val_transform(self, size):
        additional_targets = {
            f"image_{i+1}": f"image_{i+1}" for i in range(self.sequence_length - 1)
        }
        return Compose(
            [
                IsotropicResize(
                    max_side=size,
                    interpolation_down=cv2.INTER_AREA,
                    interpolation_up=cv2.INTER_CUBIC,
                ),
                PadIfNeeded(
                    min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT
                ),
            ],
            additional_targets=additional_targets,
        )

    def resize_with_pad(
        self,
        image,
        new_shape,
        padding_color: Tuple[int] = (255, 255, 255),
    ) -> np.array:
        """Maintains aspect ratio and resizes with padding.
        Params:
            image: Image to be resized.
            new_shape: Expected (width, height) of new image.
            padding_color: Tuple in BGR of padding color
        Returns:
            image: Resized image with padding
        """
        original_shape = (image.shape[1], image.shape[0])
        ratio = float(max(new_shape)) / max(original_shape)
        new_size = tuple([int(x * ratio) for x in original_shape])
        image = cv2.resize(image, new_size)
        delta_w = new_shape[0] - new_size[0]
        delta_h = new_shape[1] - new_size[1]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        image = cv2.copyMakeBorder(
            image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color
        )
        return image

    def __getitem__(self, index):

        video_dir = self.video_dirs[index]
        video_name = os.path.basename(video_dir) + ".mp4"
        label = self.labels[self.labels.iloc[:, 0] == video_name]["label"].values[0]

        # Obtener los frames de la subcarpeta
        frames = []

        # aplicar transformaciones a nivel de frame
        if self.mode == "train":
            transform = self.create_train_transforms(self.image_size)
        else:
            transform = self.create_val_transform(self.image_size)

        ###########################################################################
        # for frame_file in sorted(os.listdir(video_dir)):
        #     frame_path = os.path.join(video_dir, frame_file)
        #     frame = cv2.imread(frame_path)
        #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convertir BGR a RGB
        #     frame = cv2.resize(frame, (self.image_size, self.image_size))
        #     frames.append(frame)

        # # sequence = frames[: self.sequence_length]
        # # Crear una secuencia de frames aleatoria de tamaño sequence length
        # if len(frames) >= self.sequence_length:
        #     start_index = np.random.randint(0, len(frames) - self.sequence_length + 1)
        #     sequence = frames[start_index : start_index + self.sequence_length]

        # # Añadir padding si la secuencia es demasiado corta
        # else:
        #     sequence = frames[: self.sequence_length]
        #     sequence.extend([sequence[-1]] * (self.sequence_length - len(sequence)))
        # sequence = np.stack(sequence)

        # Obtener lista de archivos en el directorio de video

        frame_files = sorted(os.listdir(video_dir), key=self.natural_sort_key)
        # filtered_frame_files = frame_files

        filtered_frame_files = []

        for index in range(len(frame_files)):
            value = frame_files[index].split(".")[0].split("_")[1]
            if value != "1":
                filtered_frame_files.append(frame_files[index])

        # Comprobar si hay suficientes frames para una secuencia
        if len(filtered_frame_files) >= self.sequence_length:
            # Seleccionar un índice inicial aleatorio para los frames consecutivos

            #################BORRAR
            # try:
            #     for frame_file in frame_files:
            #         frame_path = os.path.join(video_dir, frame_file)
            #         frame = cv2.imread(frame_path)
            #         frame = cv2.cvtColor(
            #             frame, cv2.COLOR_BGR2RGB
            #         )  # Convertir BGR a RGB
            #         frame = cv2.resize(frame, (self.image_size, self.image_size))
            #         # sequence.append(frame)
            # except e:
            #     print(e)
            ########################

            start_index = np.random.randint(
                0, len(filtered_frame_files) - self.sequence_length + 1
            )
            selected_frame_files = filtered_frame_files[
                start_index : start_index + self.sequence_length
            ]

            # Procesar solo los frames seleccionados
            sequence = []

            for frame_file in selected_frame_files:
                frame_path = os.path.join(video_dir, frame_file)
                try:
                    frame = cv2.imread(frame_path)
                    frame = cv2.cvtColor(
                        frame, cv2.COLOR_BGR2RGB
                    )  # Convertir BGR a RGB
                    frame = cv2.resize(frame, (self.image_size, self.image_size))
                ### casos en los que existen frames vacios
                except:
                    frame_name = frame_path.split("/")[-1]
                    frame_number = int(frame_name.split("_")[0])
                    frame_index = frame_name.split("_")[1]

                    if frame_file == selected_frame_files[0]:
                        frame_number += 1
                    else:
                        frame_number -= 1
                    frame_name = str(frame_number) + "_" + frame_index

                    frame_path = os.path.join(video_dir, frame_name)
                    frame = cv2.imread(frame_path)
                    frame = cv2.cvtColor(
                        frame, cv2.COLOR_BGR2RGB
                    )  # Convertir BGR a RGB
                    frame = cv2.resize(frame, (self.image_size, self.image_size))

                sequence.append(frame)

            ##############

        else:
            # Si hay menos frames que los necesarios, procesar todos los disponibles
            sequence = []
            for frame_file in filtered_frame_files:
                frame_path = os.path.join(video_dir, frame_file)
                frame = cv2.imread(frame_path)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convertir BGR a RGB
                frame = cv2.resize(frame, (self.image_size, self.image_size))
                sequence.append(frame)

            # Añadir padding si no se alcanza el tamaño de la secuencia
            sequence.extend([sequence[-1]] * (self.sequence_length - len(sequence)))

        # Convertir la lista de frames en un array numpy
        sequence = np.stack(sequence)

        ######################################################

        # Transformacion de la secuencia
        # cv2.imwrite("preview_img.png", sequence[0])

        transformed_sequence = transform(
            image=sequence[0],
            image_1=sequence[1],
            image_2=sequence[2],
            # image_3=sequence[3],
            # image_4=sequence[4],
        )

        sequence = np.stack(
            [
                transformed_sequence["image"],
                transformed_sequence["image_1"],
                transformed_sequence["image_2"],
                # transformed_sequence["image_3"],
                # transformed_sequence["image_4"],
            ]
        )

        # cv2.imwrite("post_img.png", sequence[0])
        sequence = np.transpose(sequence, (0, 3, 1, 2))
        # Aplica transformaciones a nivel de secuencia (opcional)
        try:
            return torch.tensor(sequence).float(), torch.tensor(label).float()
        except:
            print("e")

    def natural_sort_key(self, file_name):
        return [
            int(text) if text.isdigit() else text
            for text in re.split(r"(\d+)", file_name)
        ]

    def __len__(self):
        return len(self.video_dirs)

    def get_train_counters(self):
        """
        Cuenta el número de ceros y unos en una columna específica de un DataFrame.

        Args:
            dataframe: El DataFrame de Pandas.
            columna: El nombre de la columna a analizar.

        Returns:
            Una lista con dos elementos: el número de ceros y el número de unos.
        """

        # Cuenta el número de ceros y unos utilizando value_counts() y luego convierte a una lista
        value_counts = self.labels["label"].value_counts().tolist()

        # Si solo hay una clase, agrega un cero para el valor que no está presente
        if len(value_counts) == 1:
            value_counts.append(0)

        return value_counts
