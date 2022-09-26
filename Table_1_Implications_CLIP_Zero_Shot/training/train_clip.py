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
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()
sns.set_context("talk")

sys.path.append("ANONYMOUS_ROOTDIR/develop/open-world/")
from utils import (
    svd,
    reduce_and_visualize,
    load_clip,
    encode_clip,
    encode_clip_classification,
    train_clip_toy,
    train_clip_toy_fix_init,
    ce_loss,
    uniform_loss,
    dual_ce_loss,
    simple_ce_loss,
)
from datasets import ImageCaptionDataset, ClassificationDataset


def evaluate_retrieval(image_features, text_features):
    metrics = {}
    sim = image_features @ text_features.T
    for K in [1, 5, 10]:
        pred = sim.argsort(dim=-1)
        text_r = np.mean([i in pred[i, -K:] for i in range(len(pred))])

        pred = sim.argsort(dim=0)
        image_r = np.mean([i in pred[-K:, i] for i in range(len(pred))])

        metrics[f"Text R@{K}"] = text_r
        metrics[f"Image R@{K}"] = image_r
    return metrics


def evaluate_classification(image_features, text_features, labels):
    metrics = {}
    sim = image_features @ text_features.T
    for K in [1, 5, 10]:
        pred = sim.argsort(dim=-1)
        text_r = np.mean([labels[i] in pred[i, -K:] for i in range(len(pred))])
        metrics[f"Hit@{K}"] = text_r
    return metrics


def evaluate_binary_classification(image_features, text_features, labels):
    from sklearn.metrics import roc_auc_score

    metrics = {}
    sim = image_features @ text_features.T * 100
    probs = F.softmax(sim, dim=-1)[:, 1]
    roc_auc = roc_auc_score(labels, probs)
    metrics[f"ROC-AUC"] = roc_auc
    return metrics


def move_features(image_features, text_features, evaluate_func):
    all_metrics = {}

    modality_gap = image_features.mean(axis=0) - text_features.mean(axis=0)
    modality_gap = modality_gap / modality_gap.norm()
    modality_gap.unsqueeze(0)

    for delta in np.arange(-5, 5, 0.25):
        modified_text_features = text_features + 0.5 * delta * modality_gap
        modified_text_features /= modified_text_features.norm(dim=-1, keepdim=True)

        modified_image_features = image_features - 0.5 * delta * modality_gap
        modified_image_features /= modified_image_features.norm(dim=-1, keepdim=True)

        # reduce_and_visualize(modified_image_features.numpy(), modified_text_features.numpy(), methods=['svd', 'pca'], n_dim=2)

        preds = (modified_image_features @ modified_text_features.T).argmax(dim=-1)

        gap_distance = (
            (modified_text_features.mean(axis=0) - modified_image_features.mean(axis=0))
            .norm()
            .item()
        )

        metrics = evaluate_func(modified_image_features, modified_text_features)
        all_metrics[delta] = (metrics, gap_distance, preds)

        print(delta, metrics, gap_distance)
    return all_metrics


def move_features_along_hypersphere(image_features, text_features, evaluate_func):
    return "Impossible"


def plot_metrics(all_metrics, metric_name="Hit@1"):
    xs, ys = [], []
    for delta in sorted(all_metrics.keys()):
        metrics, gap_distance, preds = all_metrics[delta]
        xs.append(gap_distance)
        ys.append(metrics[metric_name])
    print(f"Optimal {metric_name}: {max(ys)}")

    minidx = xs.index(min(xs))
    for i in range(minidx + 1, len(xs)):
        xs[i] = -xs[i]
    plt.plot(xs, ys, "o-")
    plt.xlabel("Gap Distance")
    plt.ylabel(metric_name)

    initial_gap = all_metrics[0][1]
    plt.axvline(initial_gap, color="k", linestyle="--")

    plt.show()


if __name__ == "__main__":
    temperature = int(sys.argv[1])
    print(f"Temperature: {temperature}")
    dataset = ImageCaptionDataset(split="train", max_data_size=50000)
    model = load_clip("random")
    model.logit_scale.data = torch.log(torch.tensor(temperature))
    logs, model = train_clip_toy(
        model,
        dataset,
        f"ANONYMOUS_ROOTDIR/develop/open-world/exps/random_t{temperature}_2/",
        batch_size=64,
        end_epoch=5,
    )
    logs, model = train_clip_toy_fix_init(
        model,
        dataset,
        f"ANONYMOUS_ROOTDIR/develop/open-world/exps/random_t{temperature}_fix_init/",
        batch_size=64,
        end_epoch=5,
    )
