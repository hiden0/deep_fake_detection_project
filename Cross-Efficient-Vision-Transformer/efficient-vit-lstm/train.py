from efficientnet_pytorch import EfficientNet
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from einops import rearrange, repeat
from torch import nn, einsum
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from random import random, randint, choice
from vit_pytorch import ViT
import numpy as np
from torch.optim import lr_scheduler
import os
import json
import re
from PIL import Image
from os import cpu_count
from multiprocessing.pool import Pool
from functools import partial
from torch.utils.tensorboard import SummaryWriter
from multiprocessing import Manager
from progress.bar import ChargingBar
from efficient_vit import EfficientViT
import uuid
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
import cv2
from transforms.albu import IsotropicResize
import glob
import pandas as pd
from tqdm import tqdm
from utils import get_method, check_correct, resize, shuffle_dataset, get_n_params
from sklearn.utils.class_weight import compute_class_weight
from torch.optim.lr_scheduler import LambdaLR
import collections
from deepfakes_dataset import DeepFakesDataset
import math
import yaml
import argparse


"""
##########################INN0403_CONFIG#########################


BASE_DIR = "/srv/nvme/javber/dataset/"
DATA_DIR = "/srv/nvme/javber/dataset/"
TRAINING_DIR = BASE_DIR + "train_set"
VALIDATION_DIR = BASE_DIR + "validation_set"
METADATA_PATH = os.path.join(
    TRAINING_DIR, "metadata_combinado.json"
)  # Folder containing all training metadata for DFDC dataset
VALIDATION_LABELS_PATH = os.path.join(TRAINING_DIR, "metadata.csv")
MODELS_PATH = "/srv/nvme/javber/models_save/"
#################################################################
"""

##########################INN0763_CONFIG#########################


BASE_DIR = "/srv/hdd2/javber/dataset/"
DATA_DIR = "/srv/hdd2/javber/dataset/"
TRAINING_DIR = BASE_DIR + "train_set"
VALIDATION_DIR = BASE_DIR + "validation_set"
METADATA_PATH = os.path.join(
    TRAINING_DIR, "metadata_combinado.json"
)  # Folder containing all training metadata for DFDC dataset
VALIDATION_LABELS_PATH = os.path.join(TRAINING_DIR, "metadata.csv")
MODELS_PATH = "/srv/hdd2/javber/dataset/models_save/"
#################################################################


def read_video_sequences(
    video_path, sequence_length, train_dataset, validation_dataset
):
    # Get the video label based on dataset selected
    method = get_method(video_path, DATA_DIR)
    if TRAINING_DIR in video_path:
        if "Original" in video_path:
            label = 0.0
        elif "DFDC" in video_path:
            for json_path in glob.glob(os.path.join(METADATA_PATH, "*.json")):
                with open(json_path, "r") as f:
                    metadata = json.load(f)
                video_folder_name = os.path.basename(video_path)
                video_key = video_folder_name + ".mp4"
                if video_key in metadata.keys():
                    item = metadata[video_key]
                    label = item.get("label", None)
                    if label == "FAKE":
                        label = 1.0
                    else:
                        label = 0.0
                    break
                else:
                    label = None
        else:
            label = 1.0
        if label == None:
            print("NOT FOUND", video_path)
    else:
        if "Original" in video_path:
            label = 0.0
        elif "DFDC" in video_path:
            val_df = pd.DataFrame(pd.read_csv(VALIDATION_LABELS_PATH))
            video_folder_name = os.path.basename(video_path)
            video_key = video_folder_name + ".mp4"
            label = val_df.loc[val_df["filename"] == video_key]["label"].values[0]
        else:
            label = 1.0

    # Read all frames of the video
    frames_paths = sorted(os.listdir(video_path))
    num_frames = len(frames_paths)

    # Group frames into sequences
    for i in range(0, num_frames - sequence_length + 1, sequence_length):
        sequence = []
        for j in range(sequence_length):
            frame_path = os.path.join(video_path, frames_paths[i + j])
            image = cv2.imread(frame_path)
            if image is not None:
                image_resized = cv2.resize(
                    image,
                    (config["model"]["image-size"], config["model"]["image-size"]),
                )
                sequence.append(image_resized)

        if len(sequence) == sequence_length:
            # Convert the sequence to a NumPy array and add to the dataset
            sequence_array = np.stack(sequence)
            if TRAINING_DIR in video_path:
                train_dataset.append((sequence_array, label))
            else:
                validation_dataset.append((sequence_array, label))


# Main body
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_epochs", default=300, type=int, help="Number of training epochs."
    )
    parser.add_argument(
        "--workers", default=10, type=int, help="Number of data loader workers."
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="Path to latest checkpoint (default: none).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Deepfakes",
        help="Which dataset to use (Deepfakes|Face2Face|FaceShifter|FaceSwap|NeuralTextures|All)",
    )
    parser.add_argument(
        "--max_videos",
        type=int,
        default=10,
        help="Maximum number of videos to use for training (default: all).",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Which configuration to use. See into 'config' folder.",
    )
    parser.add_argument(
        "--efficient_net",
        type=int,
        default=0,
        help="Which EfficientNet version to use (0 or 7, default: 0)",
    )
    parser.add_argument(
        "--lstm_hidden_size",
        type=int,
        default=256,
        help="Size of the lstm hidden layers",
    )
    parser.add_argument(
        "--lstm_num_layers",
        type=int,
        default=1,
        help="How many layers lstm should haver",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="How many epochs wait before stopping for validation loss not improving.",
    )

    opt = parser.parse_args()
    print(opt)
    experiment_name = opt.config.split("/")[-1]
    experiment_name = experiment_name.split(".")[0]
    writer = SummaryWriter(log_dir=os.path.join(BASE_DIR, "runs", experiment_name))

    with open(opt.config, "r") as ymlfile:
        config = yaml.safe_load(ymlfile)

    if opt.efficient_net == 0:
        channels = 1280
    else:
        channels = 2560

    model = EfficientViT(
        config=config,
        channels=channels,
        selected_efficient_net=opt.efficient_net,
        lstm_hidden_size=opt.lstm_hidden_size,
        lstm_num_layers=opt.lstm_num_layers,
    )
    ######################################
    # from torchviz import make_dot

    # x = torch.randn(1, 16, 3, 224, 224)  # batch_size=1, seq_len=16, 3 canales, 224x224
    # output = model(x)
    # dot = make_dot(output, params=dict(model.named_parameters()))
    # dot.format = "svg"
    # dot.render("efficient_vit_lstm_architecture")
    ######################################
    model.train()
    transform = transforms.ToTensor()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight-decay"],
    )
    scheduler = lr_scheduler.StepLR(
        optimizer,
        step_size=config["training"]["step-size"],
        gamma=config["training"]["gamma"],
    )
    starting_epoch = 0
    if os.path.exists(opt.resume):
        model.load_state_dict(torch.load(opt.resume))
        starting_epoch = (
            int(opt.resume.split("checkpoint")[1].split("_")[0]) + 1
        )  # The checkpoint's file name format should be "checkpoint_EPOCH"
    else:
        print("No checkpoint loaded.")

    print("Model Parameters:", get_n_params(model))

    # READ DATASET
    if opt.dataset != "All" and opt.dataset != "DFDC":
        folders = ["Original", opt.dataset]
    else:
        folders = [
            "Original",
            "DFDC",
            "Deepfakes",
            "Face2Face",
            "FaceShifter",
            "FaceSwap",
            "NeuralTextures",
        ]

    sets = [TRAINING_DIR, VALIDATION_DIR]

    paths = []
    # for dataset in sets:
    #     for folder in folders:
    #         subfolder = os.path.join(dataset, folder)
    #         for index, video_folder_name in enumerate(os.listdir(subfolder)):
    #             if index == opt.max_videos:
    #                 break

    #             if os.path.isdir(os.path.join(subfolder, video_folder_name)):
    #                 paths.append(os.path.join(subfolder, video_folder_name))

    # mgr = Manager()
    # train_dataset = mgr.list()
    # validation_dataset = mgr.list()

    train_dataset = DeepFakesDataset(
        data_root=TRAINING_DIR,
        labels_csv=VALIDATION_LABELS_PATH,
        image_size=config["model"]["image-size"],
        sequence_length=10,
        mode="train",
    )
    train_samples = train_dataset.__len__()
    train_counters = train_dataset.get_train_counters()
    class_weights = train_counters[0] / train_counters[1]
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([class_weights]))

    dl = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["training"]["bs"],
        shuffle=True,
        sampler=None,
        batch_sampler=None,
        num_workers=opt.workers,
        collate_fn=None,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
        # prefetch_factor=None,
        prefetch_factor=4,
        persistent_workers=False,
    )
    del train_dataset
    """
    for batch_idx, data in enumerate(dl):
        # Aquí, 'data' contiene un batch de datos (inputs y etiquetas)
        print("Checking data loader...")
        print(f"Batch {batch_idx+1}/{len(dl)}")
        # Por ejemplo, puedes imprimir las dimensiones de los datos:
        # print(data.shape)
    """

    validation_dataset = DeepFakesDataset(
        data_root=VALIDATION_DIR,
        labels_csv=VALIDATION_LABELS_PATH,
        image_size=config["model"]["image-size"],
        sequence_length=10,
        mode="val",
    )

    validation_samples = validation_dataset.__len__()

    val_dl = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=config["training"]["bs"],
        shuffle=True,
        sampler=None,
        batch_sampler=None,
        num_workers=opt.workers,
        collate_fn=None,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
        prefetch_factor=4,
        # prefetch_factor=None,
        persistent_workers=False,
    )
    del validation_dataset

    # for batch_idx, data in enumerate(val_dl):
    #     # Aquí, 'data' contiene un batch de datos (inputs y etiquetas)
    #     print("Checking data loader...")
    #     print(f"Batch {batch_idx+1}/{len(val_dl)}")
    #     # Por ejemplo, puedes imprimir las dimensiones de los datos:
    #     # print(data.shape)

    model = model.cuda()
    counter = 0
    not_improved_loss = 0
    previous_loss = math.inf
    for t in range(starting_epoch, opt.num_epochs + 1):
        if not_improved_loss == opt.patience:
            break
        counter = 0

        total_loss = 0
        total_val_loss = 0
        writer.add_scalar("Learning Rate", scheduler.get_last_lr()[0], t)
        bar = ChargingBar(
            "EPOCH #" + str(t), max=(len(dl) * config["training"]["bs"]) + len(val_dl)
        )
        train_correct = 0
        positive = 0
        negative = 0
        # images = np.transpose(images, (0, 3, 1, 2))
        for index, (images, labels) in enumerate(dl):
            # no transponemos nada, las dims son (batch_size, nframes, width, height, nchannels)
            # print(f"Iteration: {index}/{len(dl)}")
            labels = labels.unsqueeze(1)
            images = images.cuda()
            # print(f"labels= {labels}")

            y_pred = model(images)
            y_pred = y_pred.cpu()

            loss = loss_fn(y_pred, labels)

            corrects, positive_class, negative_class = check_correct(y_pred, labels)
            train_correct += corrects
            positive += positive_class
            negative += negative_class
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()
            counter += 1
            total_loss += round(loss.item(), 2)

            if index % 50 == 0:  # Intermediate metrics print
                print(f"Iteration: {index}/{len(dl)}")
                print(
                    "\nLoss: ",
                    total_loss / counter,
                    "Accuracy: ",
                    train_correct / (counter * config["training"]["bs"]),
                    "Train 0s: ",
                    negative,
                    "Train 1s:",
                    positive,
                )

            for i in range(config["training"]["bs"]):
                bar.next()

        val_correct = 0
        val_positive = 0
        val_negative = 0
        val_counter = 0
        all_val_labels = []
        all_val_preds = []
        train_correct /= train_samples
        total_loss /= counter

        # Registro de las métricas de entrenamiento en TensorBoard
        writer.add_scalar("Loss/Train", total_loss, t)
        writer.add_scalar("Accuracy/Train", train_correct, t)
        torch.cuda.empty_cache()

        for index, (val_images, val_labels) in enumerate(val_dl):

            # val_images = np.transpose(val_images, (0, 3, 1, 2))

            val_images = val_images.cuda()
            val_labels = val_labels.unsqueeze(1)
            val_pred = model(val_images)
            with torch.no_grad():
                val_pred = val_pred.cpu()
                val_loss = loss_fn(val_pred, val_labels)
                total_val_loss += round(val_loss.item(), 2)
                corrects, positive_class, negative_class = check_correct(
                    val_pred, val_labels
                )
                # Guardar las predicciones y etiquetas verdaderas para calcular métricas después
                all_val_labels.extend(val_labels.cpu().numpy())
                all_val_preds.extend(torch.sigmoid(val_pred).cpu().numpy())
                val_correct += corrects
                val_positive += positive_class
                val_counter += 1
                val_negative += negative_class
                bar.next()

        scheduler.step()
        bar.finish()

        # Después de la validación, calcular las métricas
        all_val_labels = np.array(all_val_labels).astype(int)
        all_val_preds = np.array(all_val_preds).round().astype(int)

        # Calcular F1 score
        f1 = f1_score(all_val_labels, all_val_preds)

        # Calcular AUC (Área bajo la curva ROC)
        #        auc = roc_auc_score(all_val_labels, all_val_preds)

        # Calcular la matriz de confusión
        tn, fp, fn, tp = confusion_matrix(all_val_labels, all_val_preds).ravel()

        # Tasa de falsos positivos (FPR) y tasa de falsos negativos (FNR)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

        # Registrar las métricas en TensorBoard
        # writer.add_scalar("Loss/Validation", total_val_loss / val_counter, t)
        # writer.add_scalar("Accuracy/Validation", val_correct / validation_samples, t)
        writer.add_scalar("Loss/Validation", total_val_loss, t)
        writer.add_scalar("Accuracy/Validation", val_correct, t)
        writer.add_scalar("F1_Score/Validation", f1, t)
        #        writer.add_scalar("AUC/Validation", auc, t)
        writer.add_scalar("FPR/Validation", fpr, t)
        writer.add_scalar("FNR/Validation", fnr, t)
        total_val_loss /= val_counter
        val_correct /= validation_samples

        # writer.add_scalar("Loss/Validation", total_val_loss, t)
        # writer.add_scalar("Accuracy/Validation", val_correct, t)
        if previous_loss <= total_val_loss:
            print("Validation loss did not improved")
            not_improved_loss += 1
        else:
            not_improved_loss = 0

        previous_loss = total_val_loss
        print(
            "#"
            + str(t)
            + "/"
            + str(opt.num_epochs)
            + " loss:"
            + str(total_loss)
            + " accuracy:"
            + str(train_correct)
            + " val_loss:"
            + str(total_val_loss)
            + " val_accuracy:"
            + str(val_correct)
            + " val_0s:"
            + str(val_negative)
            + "/"
            # + str(np.count_nonzero(validation_labels == 0))
            + " val_1s:"
            + str(val_positive)
            + "/"
            # + str(np.count_nonzero(validation_labels == 1))
        )

        if not os.path.exists(MODELS_PATH):
            os.makedirs(MODELS_PATH)

        torch.save(
            model.state_dict(),
            os.path.join(
                MODELS_PATH,
                experiment_name,
                "_checkpoint_" + str(t),
            ),
        )
    writer.close()
