# Most commonly used
import sys
import os
import json
import pickle
import math
from collections import Counter, defaultdict
from functools import partial
from tqdm import tqdm, trange
from colors import blue, red, green, cyan

# Numerical computation
import numpy as np
import torch
import torch.nn.functional as F

# Visualization
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap.umap_ import UMAP
from sklearn.cluster import KMeans

# Density estimation
sys.path.append("ANONYMOUS_ROOTDIR/develop/open-world/vonmiseskde")
from vonmiseskde import VonMisesKDE
from sklearn.neighbors import KernelDensity

# Image processing
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# Multimodal model
sys.path.append("ANONYMOUS_ROOTDIR/develop/open-world/CLIP")
import clip
from clip.model import CLIP


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_clip(model_path=None):
    device = get_device()
    if model_path is None:
        print("Loading original model...")
        model, _ = clip.load("ViT-B/16", device=device)
        model.float()
    else:
        print(f"Loading model from {model_path}...")
        model = CLIP(
            embed_dim=512,
            image_resolution=224,
            vision_layers=12,
            vision_width=768,
            vision_patch_size=16,
            context_length=77,
            vocab_size=49408,
            transformer_width=512,
            transformer_heads=8,
            transformer_layers=12,
        ).to(device)
        if model_path != "random":
            model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"Temperature: {model.logit_scale.exp()}")
    return model


def encode_clip(model, dataset, batch_size=32):
    device = get_device()

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=batch_size // 4,
        collate_fn=dataset.collate_fn,
    )

    all_image_features, all_text_features = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            image_inputs, text_inputs = batch
            image_inputs, text_inputs = image_inputs.to(device), text_inputs.to(device)

            image_features = model.encode_image(image_inputs).cpu()
            image_features /= image_features.norm(dim=-1, keepdim=True)
            all_image_features.append(image_features)

            text_features = model.encode_text(text_inputs).cpu()
            text_features /= text_features.norm(dim=-1, keepdim=True)
            all_text_features.append(text_features)

        all_image_features = torch.cat(all_image_features, dim=0)
        all_text_features = torch.cat(all_text_features, dim=0)

    return all_image_features, all_text_features


def align_loss_(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()


def uniform_loss_(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()


def ce_loss(model, image_features, text_features):
    loss_func = torch.nn.CrossEntropyLoss()

    logit_scale = model.logit_scale.exp()
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()

    batch_size = image_features.size(0)
    device = get_device()
    ground_truth = torch.arange(batch_size, dtype=torch.long, device=device)

    loss = (
        loss_func(logits_per_image, ground_truth)
        + loss_func(logits_per_text, ground_truth)
    ) / 2
    return loss


def uniform_loss(model, image_features, text_features):
    loss = (uniform_loss_(image_features) + uniform_loss_(text_features)) / 2
    return loss


def dual_ce_loss(model, image_features, text_features):
    loss_func = torch.nn.CrossEntropyLoss()

    features = torch.cat([image_features, text_features], 0)
    sims = features @ features.t()

    logit_scale = model.logit_scale.exp()
    logits = sims * logit_scale

    batch_size = image_features.size(0)
    logits_per_image = logits[:batch_size, :].contiguous()
    logits_per_image[torch.arange(batch_size), torch.arange(batch_size)] -= 10000
    logits_per_text = logits[batch_size:, :].contiguous()
    logits_per_text[
        torch.arange(batch_size), torch.arange(batch_size) + batch_size
    ] -= 10000

    device = get_device()
    image_ground_truth = (
        torch.arange(batch_size, dtype=torch.long, device=device) + batch_size
    )
    text_ground_truth = torch.arange(batch_size, dtype=torch.long, device=device)

    loss = (
        loss_func(logits_per_image, image_ground_truth)
        + loss_func(logits_per_text, text_ground_truth)
    ) / 2
    return loss


def simple_ce_loss(model, image_features, text_features):
    loss_func = torch.nn.CrossEntropyLoss(reduction="none")

    logit_scale = model.logit_scale.exp()
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()

    preds_per_image = torch.argmax(logits_per_image, dim=1)
    preds_per_text = torch.argmax(logits_per_text, dim=1)

    batch_size = image_features.size(0)
    device = get_device()
    ground_truth = torch.arange(batch_size, dtype=torch.long, device=device)

    correct_per_image = (preds_per_image == ground_truth).float()
    correct_per_text = (preds_per_text == ground_truth).float()

    loss_img = (loss_func(logits_per_image, ground_truth) * correct_per_image).sum() / (
        correct_per_image.sum() + 1e-6
    )
    loss_text = (loss_func(logits_per_text, ground_truth) * correct_per_text).sum() / (
        correct_per_text.sum() + 1e-6
    )

    loss = (loss_img + loss_text) / 2
    return loss


def train_clip_toy_fix_init(
    model,
    dataset,
    model_path,
    batch_size=32,
    start_epoch=0,
    end_epoch=10,
    loss_funcs=[ce_loss],
):
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    device = get_device()

    if start_epoch == 0:
        print("Training original model...")
        torch.save(model.state_dict(), f"{model_path}/model_epoch_{start_epoch}.pt")
    else:
        print(f"Loading model from {model_path} and continue training...")
        assert os.path.exists(f"{model_path}/model_epoch_{start_epoch}.pt")
        model.load_state_dict(torch.load(f"{model_path}/model_epoch_{start_epoch}.pt"))

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=batch_size // 4,
        collate_fn=dataset.collate_fn,
        drop_last=True,
    )

    all_image_features, all_text_features = encode_clip(model, dataset)
    yx = all_image_features.t() @ all_text_features
    u, s, v = torch.svd(yx)
    w = u @ v.T
    torch.save([w, all_image_features, all_text_features], f"{model_path}/w.pt")
    all_text_features_transform = all_text_features @ w.T
    w = w.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-5, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2
    )

    logs = {}
    for epoch in range(start_epoch + 1, end_epoch + 1):
        logs[epoch] = []
        bar = tqdm(dataloader)
        for i, batch in enumerate(bar):
            image_inputs, text_inputs = batch
            image_inputs, text_inputs = image_inputs.to(device), text_inputs.to(device)

            image_features = model.encode_image(image_inputs)
            text_features = model.encode_text(text_inputs)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features = text_features @ w.T

            losses = [
                loss_func(model, image_features, text_features)
                for loss_func in loss_funcs
            ]
            loss = sum(losses)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logs[epoch].append(
                {"loss": loss.item(), "losses": [loss.item() for loss in losses]}
            )
            bar.set_description(f"Epoch {epoch}/{end_epoch}, Loss: {logs[epoch][i]}")

        torch.save(model.state_dict(), f"{model_path}/model_epoch_{epoch}.pt")

        epoch_loss = np.mean([item["loss"] for item in logs[epoch]])
        epoch_losses = [
            np.mean([item["losses"][i] for item in logs[epoch]])
            for i in range(len(loss_funcs))
        ]
        print(f"Epoch {epoch}: loss = {epoch_loss:.4f}, losses = {epoch_losses}")
    return model, logs


def train_clip_toy(
    model,
    dataset,
    model_path,
    batch_size=32,
    start_epoch=0,
    end_epoch=10,
    loss_funcs=[ce_loss],
):
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    device = get_device()

    if start_epoch == 0:
        print("Training original model...")
        torch.save(model.state_dict(), f"{model_path}/model_epoch_{start_epoch}.pt")
    else:
        print(f"Loading model from {model_path} and continue training...")
        assert os.path.exists(f"{model_path}/model_epoch_{start_epoch}.pt")
        model.load_state_dict(torch.load(f"{model_path}/model_epoch_{start_epoch}.pt"))

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=batch_size // 4,
        collate_fn=dataset.collate_fn,
        drop_last=True,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-5, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2
    )

    logs = {}
    for epoch in range(start_epoch + 1, end_epoch + 1):
        logs[epoch] = []
        bar = tqdm(dataloader)
        for i, batch in enumerate(bar):
            image_inputs, text_inputs = batch
            image_inputs, text_inputs = image_inputs.to(device), text_inputs.to(device)

            image_features = model.encode_image(image_inputs)
            text_features = model.encode_text(text_inputs)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            losses = [
                loss_func(model, image_features, text_features)
                for loss_func in loss_funcs
            ]
            loss = sum(losses)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logs[epoch].append(
                {"loss": loss.item(), "losses": [loss.item() for loss in losses]}
            )
            bar.set_description(f"Epoch {epoch}/{end_epoch}, Loss: {logs[epoch][i]}")

        torch.save(model.state_dict(), f"{model_path}/model_epoch_{epoch}.pt")

        epoch_loss = np.mean([item["loss"] for item in logs[epoch]])
        epoch_losses = [
            np.mean([item["losses"][i] for item in logs[epoch]])
            for i in range(len(loss_funcs))
        ]
        print(f"Epoch {epoch}: loss = {epoch_loss:.4f}, losses = {epoch_losses}")
    return model, logs


def encode_clip_classification(
    model, dataset, prompt="a photo of a {}.", batch_size=32
):
    device = get_device()

    text_inputs = torch.cat(
        [clip.tokenize(prompt.format(c)) for c in dataset.data.classes]
    ).to(device)
    with torch.no_grad():
        all_text_features = model.encode_text(text_inputs).cpu()
        all_text_features /= all_text_features.norm(dim=-1, keepdim=True)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=batch_size // 4,
        collate_fn=dataset.collate_fn,
    )

    all_image_features = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            image_inputs, labels = batch
            image_inputs = image_inputs.to(device)

            image_features = model.encode_image(image_inputs).cpu()
            image_features /= image_features.norm(dim=-1, keepdim=True)
            all_image_features.append(image_features)

        all_image_features = torch.cat(all_image_features, dim=0)

    return all_image_features, all_text_features


def svd(X, n_components=2, return_singular_values=False):
    U, S, Vt = np.linalg.svd(X)
    X_reduce = U[:, :n_components] * S[:n_components]
    if return_singular_values:
        return X_reduce, S
    return X_reduce


def visualize_2d(clusters, colors=None, labels=None, connection=False):
    assert isinstance(clusters, list)
    for cluster in clusters:
        assert isinstance(cluster, np.ndarray)
        assert cluster.shape[1] == 2

    fig = plt.figure(figsize=(5, 5))
    if colors is None:
        colors = ["r" for i in range(len(clusters))]
    if labels is None:
        labels = [f"cluster_{i}" for i in range(len(clusters))]
    for cluster, color, label in zip(clusters, colors, labels):
        plt.scatter(cluster[:, 0], cluster[:, 1], c=color, label=label, alpha=0.2)

    if connection:
        assert len(clusters) == 2 and len(clusters[0]) == len(clusters[1])
        for i in range(len(clusters[0])):
            plt.plot(
                [clusters[0][i, 0], clusters[1][i, 0]],
                [clusters[0][i, 1], clusters[1][i, 1]],
                c="k",
                alpha=0.05,
            )
    plt.show()


def visualize_3d(clusters, colors=None, labels=None, connection=False):
    assert isinstance(clusters, list)
    assert connection == False
    for cluster in clusters:
        assert isinstance(cluster, np.ndarray)
        assert cluster.shape[1] == 3

    fig = plt.figure()
    ax = Axes3D(fig)
    if colors is None:
        colors = ["r" for i in range(len(clusters))]
    if labels is None:
        labels = [f"cluster_{i}" for i in range(len(clusters))]
    for cluster, color, label in zip(clusters, colors, labels):
        ax.scatter(
            cluster[:, 0], cluster[:, 1], cluster[:, 2], c=color, label=label, alpha=0.2
        )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    fig.add_axes(ax)
    plt.show()


def dim_reduce(features, n_dim=2, methods=["svd", "pca", "tsne", "umap"]):
    assert isinstance(features, np.ndarray)

    features_reduce = {}
    for method in methods:
        if method == "svd":
            features_reduce[method] = svd(features, n_components=n_dim)
        else:
            projector = eval(method.upper())(n_components=n_dim)
            features_reduce[method] = projector.fit_transform(features)
    return features_reduce


def reduce_and_visualize(
    image_features,
    text_features,
    n_dim=2,
    methods=["svd", "pca", "tsne", "umap"],
    connection=False,
):
    assert isinstance(image_features, np.ndarray) and isinstance(
        text_features, np.ndarray
    )
    assert n_dim in [2, 3]

    features = np.concatenate([image_features, text_features], axis=0)
    features_reduce = dim_reduce(features, n_dim=n_dim, methods=methods)

    for i, method in enumerate(methods):
        image_features_reduce = features_reduce[method][: len(image_features)]
        text_features_reduce = features_reduce[method][len(image_features) :]
        eval(f"visualize_{n_dim}d")(
            [image_features_reduce, text_features_reduce],
            colors=["r", "b"],
            connection=connection,
        )


def convert_image_to_rgb(image):
    return image.convert("RGB")


def estimate_density(image_features, text_features):
    x_plot = np.linspace(-1.2, 1.2, 100)
    y_plot = np.linspace(-1.2, 1.2, 100)
    xy_plot = np.array(np.meshgrid(x_plot, y_plot)).reshape(2, -1).T

    kde_image = KernelDensity(kernel="gaussian", bandwidth=0.1).fit(image_features)
    image_density = np.exp(kde_image.score_samples(xy_plot))

    kde_text = KernelDensity(kernel="gaussian", bandwidth=0.1).fit(text_features)
    text_density = np.exp(kde_text.score_samples(xy_plot))

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(
        image_density.reshape(100, 100),
        extent=(-1.2, 1.2, -1.2, 1.2),
        origin="lower",
        cmap="Reds",
        alpha=0.5,
        vmin=min([image_density.min(), text_density.min()]),
        vmax=max([image_density.max(), text_density.max()]),
    )
    plt.scatter(image_features[:, 0], image_features[:, 1], c="red", alpha=0.05)

    plt.subplot(1, 2, 2)
    plt.imshow(
        text_density.reshape(100, 100),
        extent=(-1.2, 1.2, -1.2, 1.2),
        origin="lower",
        cmap="Blues",
        alpha=0.5,
        vmin=min([image_density.min(), text_density.min()]),
        vmax=max([image_density.max(), text_density.max()]),
    )
    plt.scatter(text_features[:, 0], text_features[:, 1], c="blue", alpha=0.05)

    print(
        text_density.min(),
        text_density.max(),
        text_density.mean(),
        image_density.min(),
        image_density.max(),
        image_density.mean(),
    )


def estimate_angle_density(image_features, text_features):
    image_features_angle = [
        np.arctan2(image_features[i, 1], image_features[i, 0]).item()
        for i in range(len(image_features))
    ]
    text_features_angle = [
        np.arctan2(text_features[i, 1], text_features[i, 0]).item()
        for i in range(len(text_features))
    ]

    kappa = 25
    kde_image = VonMisesKDE(image_features_angle, weights=[], kappa=kappa)
    kde_text = VonMisesKDE(text_features_angle, weights=[], kappa=kappa)

    test_x = np.linspace(-math.pi, math.pi, 100)

    # # Display individual distributions
    # for i in np.arange(0, len(text_features_angle)):
    #     sample = text_features_angle[i]
    #     test_y = kde_text.vonMisesPDF(test_x, sample)
    #     test_y = test_y / test_y.sum()
    #     plt.plot(test_x, test_y, color='gray', alpha=0.5)

    # Display posterior estimate
    plt.figure(figsize=(10, 1))

    plt.subplot(1, 2, 1)
    plt.plot(test_x, kde_image.evaluate(test_x), zorder=20, color="red", alpha=0.5)
    plt.fill_between(
        test_x, kde_image.evaluate(test_x), step="pre", alpha=0.2, color="red"
    )
    plt.xlim(-math.pi, math.pi)
    plt.ylim(0, 1)

    plt.subplot(1, 2, 2)
    plt.plot(test_x, kde_text.evaluate(test_x), zorder=20, color="blue", alpha=0.5)
    plt.fill_between(
        test_x, kde_text.evaluate(test_x), step="pre", alpha=0.2, color="blue"
    )
    plt.xlim(-math.pi, math.pi)
    plt.ylim(0, 1)


if __name__ == "__main__":
    ##### Test svd() #####
    X = np.arange(100).reshape(10, 10)
    X_2d = svd(X)
    assert X_2d.shape == (10, 2)
