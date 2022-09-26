import sys
import os
import json
import pickle
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.append("ANONYMOUS_ROOTDIR/develop/open-world/vision")
from torchvision import transforms
from torchvision.datasets import (
    CIFAR10,
    CIFAR100,
    MNIST,
    ImageNet,
    SVHN,
    Flowers102,
    EuroSAT,
    ImageFolder,
)

sys.path.append("ANONYMOUS_ROOTDIR/develop/open-world/CLIP")
import clip


def get_default_transform():
    return clip.load("ViT-B/16", device="cpu")[1]


def get_default_tokenizer():
    return clip.tokenize


class ImageCaptionDataset(Dataset):
    def __init__(
        self,
        base_dir="ANONYMOUS_ROOTDIR/data/COCO/",
        split="val",
        transform=None,
        tokenizer=None,
        max_data_size=None,
    ):
        self.base_dir = base_dir
        self.split = split
        self.transform = transform if transform is not None else get_default_transform()
        self.tokenizer = tokenizer if tokenizer is not None else get_default_tokenizer()
        self.max_data_size = max_data_size

        data = json.load(open(f"{self.base_dir}/annotations/captions_{split}2017.json"))
        id2file = {item["id"]: item["coco_url"] for item in data["images"]}
        id2caption = {item["image_id"]: item["caption"] for item in data["annotations"]}
        self.data = [
            (id2file[id].replace("http://images.cocodataset.org/", ""), id2caption[id])
            for id in id2caption
        ]

        if self.max_data_size is not None:
            np.random.seed(1234)
            indices = np.random.choice(
                len(self.data), size=max_data_size, replace=False
            )
            self.data = [self.data[i] for i in indices]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filename, caption = self.data[idx]
        im = Image.open(f"{self.base_dir}/{filename}")
        image_input = self.transform(im)
        if len(image_input.shape) == 4:
            image_input = image_input[0]
        text_input = self.tokenizer(caption)
        if len(text_input.shape) == 2:
            text_input = text_input[0]
        return image_input, text_input

    @staticmethod
    def collate_fn(batch):
        images, texts = zip(*batch)
        images = torch.stack(images, dim=0)
        texts = torch.stack(texts, dim=0)
        return images, texts


class ClassificationDataset(Dataset):
    def __init__(self, name="CIFAR100", transform=None, max_data_size=None):
        self.name = name
        self.transform = transform if transform is not None else get_default_transform()
        self.max_data_size = max_data_size

        if self.name in ["CIFAR100", "CIFAR10", "MNIST"]:
            self.data = eval(name)(
                root=os.path.expanduser("~/.cache"), download=True, train=False
            )
        elif self.name in ["ImageNet"]:
            self.data = eval(name)(root=os.path.expanduser("~/.cache"), split="val")
            # if self.name == "ImageNet":
            #     self.data.classes = json.load(open(os.path.expanduser('~/.cache/imagenet_classes.json')))
        elif self.name in ["ImageNetSketch", "HateSpeechMeme"]:
            self.data = ImageFolder(
                root=f"ANONYMOUS_ROOTDIR/data/{self.name}/imagefolder"
            )
            if self.name == "ImageNetSketch":
                lines = [
                    line.strip().split()
                    for line in open(
                        f"ANONYMOUS_ROOTDIR/data/{self.name}/map_clsloc.txt"
                    )
                ]
                for line in lines:
                    assert len(line) == 3
                mapping = {line[0]: line[2].replace("_", " ") for line in lines}
                self.data.classes = [mapping[id] for id in self.data.classes]
        elif self.name in ["SVHN", "Flowers102"]:
            self.data = eval(name)(
                root=os.path.expanduser("~/.cache"), download=True, split="test"
            )
            if self.name == "SVHN":
                self.data.classes = [i for i in range(10)]
            elif self.name == "Flowers102":
                self.data.classes = json.load(
                    open(os.path.expanduser("~/.cache/flowers-102/mapping.json"))
                )
                self.data._labels = [i - 1 for i in self.data._labels]
        elif self.name in ["EuroSAT"]:
            self.data = eval(name)(root=os.path.expanduser("~/.cache"), download=True)
            if self.name == "EuroSAT":
                self.data.classes = json.load(
                    open(os.path.expanduser("~/.cache/eurosat/mapping.json"))
                )
        else:
            raise ValueError(f"Unknown dataset: {self.name}")

        if self.max_data_size is not None:
            raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        im, label = self.data[idx]
        image_input = self.transform(im)
        if len(image_input.shape) == 4:
            image_input = image_input[0]
        return image_input, label

    @staticmethod
    def collate_fn(batch):
        images, labels = zip(*batch)
        images = torch.stack(images, dim=0)
        labels = torch.tensor(labels)
        return images, labels


if __name__ == "__main__":
    dataset = ImageCaptionDataset()
    print(dataset[0])
    dataset = ClassificationDataset()
    print(dataset[0])
