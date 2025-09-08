import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss


def train_epoch(
    net: torch.nn,
    dataloader: DataLoader,
    optimizer: Optimizer,
    criterion: _Loss,
    device: str,
    grad_clip=5.0,
) -> float:

    net.train()

    losses = []
    for image_tensor, label in dataloader:

        image_tensor = image_tensor.to(device)
        target = label.to(device).to(torch.float)

        logit = net(image_tensor)
        loss = criterion(logit, target)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
        optimizer.step()

        losses.extend([loss.item()] * image_tensor.size(0))

        # ####################################################
        # # show images on tensorboard for debugging purpose #
        # ####################################################
        # import os
        # import cv2
        # dir_save = "input_checks"
        # os.makedirs(dir_save, exist_ok=True)
        # for i in range(len(target)):
        #     img = (image_tensor[i][0].cpu().numpy()*255).astype(np.uint8)
        #     l = target[i]
        #     cv2.imwrite(os.path.join(dir_save, f"index_{i}_label_{l}.png"), img)
        # exit(1)

    mean_loss = np.mean(losses)

    return mean_loss
