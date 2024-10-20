from efficientnet_pytorch import EfficientNet
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from einops import rearrange, repeat
from torch import nn, einsum
import torch.nn as nn
import torch.nn.functional as F
from vit_pytorch import ViT
import numpy as np
from torch.optim import lr_scheduler
import os
from torch.utils.tensorboard import SummaryWriter
from efficient_vit import EfficientViT
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.metrics import accuracy_score, roc_curve, det_curve
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
from utils import get_method, check_correct_test, resize, shuffle_dataset, get_n_params
from deepfakes_dataset import DeepFakesDataset
import yaml
import argparse
import seaborn as sns
import matplotlib.pyplot as plt


##########################INN0763_CONFIG#########################

BASE_DIR = "/srv/hdd2/javber/dataset/"
DATA_DIR = "/srv/hdd2/javber/dataset/"
TRAINING_DIR = BASE_DIR + "train_set"
VALIDATION_DIR = BASE_DIR + "test_set"
METADATA_PATH = os.path.join(
    TRAINING_DIR, "metadata_combinado.json"
)  # Folder containing all training metadata for DFDC dataset
VALIDATION_LABELS_PATH = os.path.join(BASE_DIR, "metadata_final.csv")
MODELS_PATH = "/srv/hdd2/javber/dataset/models_save/"

#################################################################

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
    writer = SummaryWriter(log_dir=os.path.join(BASE_DIR, "tests", experiment_name))

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

    if os.path.exists(opt.resume):
        model.load_state_dict(torch.load(opt.resume))

    else:
        print("No checkpoint loaded.")

    print("Model Parameters:", get_n_params(model))

    validation_dataset = DeepFakesDataset(
        data_root=VALIDATION_DIR,
        labels_csv=VALIDATION_LABELS_PATH,
        image_size=config["model"]["image-size"],
        sequence_length=5,
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

    model = model.cuda()

    val_correct = 0
    val_positive = 0
    val_negative = 0
    val_counter = 0
    all_val_labels = []
    all_val_preds = []
    umbral = 0.95
    for index, (val_images, val_labels) in enumerate(val_dl):

        # val_images = np.transpose(val_images, (0, 3, 1, 2))
        val_images = val_images.cuda()
        val_labels = val_labels.unsqueeze(1)
        val_pred = model(val_images)
        with torch.no_grad():

            val_pred = val_pred.cpu()
            corrects, positive_class, negative_class = check_correct_test(
                val_pred, val_labels, umbral
            )
            val_pred = torch.sigmoid(val_pred)
            all_val_labels.extend(val_labels.cpu().numpy())
            all_val_preds.extend(val_pred.cpu().numpy())
            val_correct += corrects
            val_positive += positive_class
            val_counter += 1
            val_negative += negative_class

    #####################################################################################
    ###################################ROC / DET CURVES##################################
    #####################################################################################
    # Obtener las probabilidades en lugar de las predicciones hard para la curva ROC
    all_val_labels = np.array(all_val_labels).astype(int)
    all_val_preds_float = all_val_preds
    all_val_preds_prob = np.array(all_val_preds_float)

    # Calcular la curva ROC
    fpr, tpr, thresholds = roc_curve(all_val_labels, all_val_preds_prob)
    auc = roc_auc_score(all_val_labels, all_val_preds_prob)
    print("AUC: ", auc)
    # Graficar la curva ROC
    plt.figure(figsize=(6, 6), dpi=300)
    plt.plot(fpr, tpr, color="blue", label="ROC curve")
    plt.plot([0, 1], [0, 1], color="red", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend()
    plt.grid()
    # Añadir a TensorBoard
    writer.add_figure("ROC Curve", plt.gcf(), 1)
    plt.close()

    # Calcular la curva DET
    fpr_det, fnr_det, thresholds_det = det_curve(all_val_labels, all_val_preds_prob)

    # Graficar la curva DET
    plt.figure(figsize=(6, 6), dpi=300)
    plt.plot(fpr_det, fnr_det, color="green", label="DET curve")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("False Negative Rate (FNR)")
    plt.title("Detection Error Tradeoff (DET) Curve")
    plt.legend()
    plt.grid()
    # Añadir a TensorBoard
    writer.add_figure("DET Curve", plt.gcf(), 1)
    plt.close()

    #####################################################################################
    ###################################THRESHOLD OPTIM##################################
    #####################################################################################

    # Calcular FPR, TPR, y umbrales
    # fpr, tpr, thresholds = roc_curve(all_val_labels_, all_val_preds)

    # Método 1: Distancia mínima al punto (0, 1)
    distancias = np.sqrt(fpr**2 + (1 - tpr) ** 2)
    indice_umbral_optimo_distancia = np.argmin(distancias)
    umbral_optimo_distancia = thresholds[indice_umbral_optimo_distancia]

    # Método 2: Maximizar Youden's J
    # print(f"tpr: {tpr} , fpr: {fpr}")
    youden_j = tpr - fpr
    indice_umbral_optimo_j = np.argmax(youden_j)
    umbral_optimo_j = thresholds[indice_umbral_optimo_j]

    print(f"Umbral óptimo (Distancia mínima a (0,1)): {umbral_optimo_distancia}")
    print(f"Umbral óptimo (Max Youden's J): {umbral_optimo_j}")

    #####################################################################################
    ###################################STANDARD METRICS##################################
    #####################################################################################

    # IMPORTANTE, MODIFICA SIGUIENDO EL UMBRAL OPTIMO O SIMPLE REDONDEO
    all_val_preds = np.array(all_val_preds)
    all_val_preds = (all_val_preds >= umbral).astype(int)

    # all_val_preds = np.array(all_val_preds).round().astype(int)

    # Calcular F1 score
    f1 = f1_score(all_val_labels, all_val_preds)

    # Calcular la matriz de confusión
    tn, fp, fn, tp = confusion_matrix(all_val_labels, all_val_preds).ravel()
    conf_matrix = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    ax.set_title("Confusion Matrix")

    # Añadir la figura a TensorBoard
    writer.add_figure("Confusion Matrix", fig, 1)

    # Cerrar la figura para liberar memoria
    plt.close(fig)
    precision = precision_score(all_val_labels, all_val_preds)
    recall = recall_score(all_val_labels, all_val_preds)

    # Tasa de falsos positivos (FPR) y tasa de falsos negativos (FNR)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    writer.add_scalar("Accuracy/Validation", val_correct / validation_samples, 1)
    writer.add_scalar("Precision/Validation", precision, 1)  # NUEVA LÍNEA
    writer.add_scalar("Recall/Validation", recall, 1)  # NUEVA LÍNE
    writer.add_scalar("F1_Score/Validation", f1, 1)
    writer.add_scalar("FPR/Validation", fpr, 1)
    writer.add_scalar("FNR/Validation", fnr, 1)
    val_correct /= validation_samples

    print(
        " val_accuracy:"
        + str(val_correct)
        + " val_0s:"
        + str(val_negative)
        + " val_1s:"
        + str(val_positive)
    )
