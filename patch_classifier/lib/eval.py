from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss


def validation_epoch(
    net: torch.nn, data_loader: DataLoader, criterion: _Loss, device: str
) -> Tuple[float, float]:

    net.eval()

    preds, labels = [], []

    losses, matches = [], []
    for image_tensor, label in data_loader:

        image_tensor = image_tensor.to(device)
        target = label.to(device).to(torch.float)

        with torch.no_grad():
            logit = net(image_tensor)
            loss = criterion(logit, target)
            losses.extend([loss.item()] * image_tensor.size(0))
            pred = (logit > 0).to(torch.long)
            matches.extend(pred.eq(target).cpu().tolist())

        import torch.nn.functional as F

        sigmoid_output = F.sigmoid(logit)
        preds.extend(sigmoid_output.cpu().tolist())
        labels.extend(label.cpu().tolist())

    mean_loss = np.mean(losses)
    mean_accuracy = np.mean(matches)

    # # debugging purpose
    # np.save("pred.npy", preds)
    # np.save("label.npy", labels)

    return mean_loss, mean_accuracy
