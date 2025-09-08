import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.datasets import ImageFolder
from PIL import Image

import albumentations as A

MEAN_CLIP = [0.48145466, 0.4578275, 0.40821073]
STD_CLIP = [0.26862954, 0.26130258, 0.27577711]


class SingleImageAugmentor:

    def __init__(self, path_augment: str):
        self.augment = A.load(path_augment)

    def __call__(self, image: Image.Image) -> np.ndarray:
        # image should be numpy array for albumentations
        img_arr = np.array(image).astype(np.uint8)
        aug_img_arr = self.augment(image=img_arr)["image"]
        return aug_img_arr


def read_rgb(path):
    return cv2.imread(path, cv2.IMREAD_COLOR)


def get_dataloader(
    root: str, path_augment: str, batch_size: int, is_train: bool, num_workers: int
) -> DataLoader:

    if path_augment is not None:
        augmentor = SingleImageAugmentor(path_augment)
        dataset = ImageFolder(
            root,
            Compose([augmentor, ToTensor(), Normalize(mean=MEAN_CLIP, std=STD_CLIP)]),
            loader=read_rgb,
        )
    else:
        dataset = ImageFolder(
            root,
            Compose([ToTensor(), Normalize(mean=MEAN_CLIP, std=STD_CLIP)]),
            loader=read_rgb,
        )

    if is_train:
        labels = [dataset.class_to_idx[c] for c in dataset.classes]
        counts = [
            sum([1 for l in dataset.targets if l == i]) for i in range(len(labels))
        ]
        weights = 1.0 / torch.tensor(counts).float()
        equal_ration_sampler = WeightedRandomSampler(
            [weights[l] for l in dataset.targets],
            len(dataset.targets),
            replacement=True,
        )
        return DataLoader(
            dataset, batch_size, sampler=equal_ration_sampler, num_workers=num_workers
        )
    else:
        return DataLoader(dataset, batch_size, shuffle=False, num_workers=num_workers)
