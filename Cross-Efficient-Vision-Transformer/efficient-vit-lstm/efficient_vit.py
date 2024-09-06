import torch
from torch import nn
from einops import rearrange
from efficientnet_pytorch import EfficientNet
import cv2
import re
import torchvision.utils as vutils
from utils import resize
import numpy as np
from torch import einsum
from random import randint


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv)

        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        attn = self.attend(dots)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Attention(
                                dim, heads=heads, dim_head=dim_head, dropout=dropout
                            ),
                        ),
                        PreNorm(
                            dim, FeedForward(dim=dim, hidden_dim=mlp_dim, dropout=0)
                        ),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class EfficientViT(nn.Module):
    def __init__(
        self,
        config,
        channels=512,
        selected_efficient_net=0,
        lstm_hidden_size=256,
        lstm_num_layers=1,
    ):
        super().__init__()

        image_size = config["model"]["image-size"]
        patch_size = config["model"]["patch-size"]
        num_classes = config["model"]["num-classes"]
        dim = config["model"]["dim"]
        depth = config["model"]["depth"]
        heads = config["model"]["heads"]
        mlp_dim = config["model"]["mlp-dim"]
        lstm_dim = config["model"]["lstm-dim"]
        emb_dim = config["model"]["emb-dim"]
        dim_head = config["model"]["dim-head"]
        dropout = config["model"]["dropout"]
        emb_dropout = config["model"]["emb-dropout"]

        assert (
            image_size % patch_size == 0
        ), "image dimensions must be divisible by the patch size"

        self.selected_efficient_net = selected_efficient_net

        if selected_efficient_net == 0:
            self.efficient_net = EfficientNet.from_pretrained("efficientnet-b0")
        else:
            self.efficient_net = EfficientNet.from_pretrained("efficientnet-b7")
            checkpoint = torch.load(
                "weights/final_999_DeepFakeClassifier_tf_efficientnet_b7_ns_0_23",
                map_location="cpu",
            )
            state_dict = checkpoint.get("state_dict", checkpoint)
            self.efficient_net.load_state_dict(
                {re.sub("^module.", "", k): v for k, v in state_dict.items()},
                strict=False,
            )

        for i in range(0, len(self.efficient_net._blocks)):
            for index, param in enumerate(self.efficient_net._blocks[i].parameters()):
                if i >= len(self.efficient_net._blocks) - 3:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        num_patches = (7 // patch_size) ** 2
        patch_dim = channels * patch_size**2

        self.patch_size = patch_size

        self.pos_embedding = nn.Parameter(torch.randn(emb_dim, 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # self.transformer = Transformer(dim, depth, heads, dim_head, lstm_dim, dropout)
        # self.to_cls_token = nn.Identity()
        # self.lstm = nn.LSTM(
        #     input_size=lstm_dim,  # Dimensión de la salida del Transformer
        #     hidden_size=lstm_hidden_size,  # Dimensión de los estados ocultos en la LSTM
        #     num_layers=lstm_num_layers,  # Número de capas LSTM
        #     batch_first=True,
        # )

        # # Capa de MLP final después de la LSTM
        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(lstm_hidden_size), nn.Linear(lstm_hidden_size, num_classes)
        # )

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.lstm = nn.LSTM(
            input_size=dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
        )

        # self.to_cls_token = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(lstm_hidden_size), nn.Linear(lstm_hidden_size, num_classes)
        )

    def forward(self, img_seq):
        p = self.patch_size

        # Asumimos que img_seq es de forma (batch_size, seq_len, channels, height, width)
        batch_size, seq_len, c, h, w = img_seq.shape
        sequence_output = []

        for t in range(seq_len):
            img = img_seq[:, t, :, :, :]
            # save_image(img[0], f"original_frame_{t}.png")
            x = self.efficient_net.extract_features(img)
            # save_embeddings(x, f"embeddings_frame_{t}")
            y = rearrange(x, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=p, p2=p)
            y = self.patch_to_embedding(y)
            cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, y), 1)
            shape = x.shape[0]
            x += self.pos_embedding[0:shape]
            x = self.dropout(x)
            x = self.transformer(x)
            sequence_output.append(x[:, 0])

        sequence_output = torch.stack(sequence_output, dim=1)

        # Procesar la secuencia con LSTM
        lstm_out, _ = self.lstm(sequence_output)

        # Toma la última salida de LSTM para la clasificación final
        out = lstm_out[:, -1, :]
        return self.mlp_head(out)

    # def forward(self, img, mask=None):
    #     p = self.patch_size
    #     x = self.efficient_net.extract_features(img)  # 1280x7x7
    #     # x = self.features(img)
    #     """
    #     for im in img:
    #         image = im.cpu().detach().numpy()
    #         image = np.transpose(image, (1,2,0))
    #         cv2.imwrite("images/image"+str(randint(0,1000))+".png", image)

    #     x_scaled = []
    #     for idx, im in enumerate(x):
    #         im = im.cpu().detach().numpy()
    #         for patch_idx, patch in enumerate(im):
    #             patch = (255*(patch - np.min(patch))/np.ptp(patch))
    #             im[patch_idx] = patch
    #             #cv2.imwrite("patches/patches_"+str(idx)+"_"+str(patch_idx)+".png", patch)
    #         x_scaled.append(im)
    #     x = torch.tensor(x_scaled).cuda()
    #     """

    #     # x2 = self.features(img)
    #     y = rearrange(x, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=p, p2=p)
    #     # y2 = rearrange(x2, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
    #     y = self.patch_to_embedding(y)
    #     cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
    #     x = torch.cat((cls_tokens, y), 1)
    #     shape = x.shape[0]
    #     x += self.pos_embedding[0:shape]
    #     x = self.dropout(x)
    #     x = self.transformer(x)
    #     x = self.to_cls_token(x[:, 0])

    #     return self.mlp_head(x)


import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np
from PIL import Image


# Función para guardar una imagen en PNG
def save_image(tensor, filename):
    img = tensor.permute(1, 2, 0).cpu().numpy()  # Cambiar de formato CHW a HWC
    img = (img * 255).astype(np.uint8)  # Escalar los valores a [0, 255]
    img = Image.fromarray(img)
    img.save(filename)


# Función para visualizar y guardar los embeddings como imagen
def save_embeddings(embeddings, filename_prefix):
    batch_size, num_embeddings, h, w = embeddings.shape

    # Vamos a seleccionar 8 embeddings del batch
    selected_embeddings = embeddings[0:8, :, :, :]  # Seleccionar 8 del batch

    # Para cada embedding seleccionado, guardamos una imagen
    for i in range(8):
        embedding = selected_embeddings[i]

        # Normalizamos el embedding para que esté entre [0, 1]
        embedding = embedding - embedding.min()
        embedding = embedding / embedding.max()

        # Guardamos cada embedding como imagen
        fig, ax = plt.subplots(figsize=(2, 2))
        ax.imshow(
            embedding[0].cpu().detach().numpy(), cmap="gray"
        )  # Selecciona solo un canal
        ax.axis("off")  # Elimina los ejes
        plt.savefig(
            f"{filename_prefix}_embedding_{i}.png", bbox_inches="tight", pad_inches=0
        )
        plt.close()
