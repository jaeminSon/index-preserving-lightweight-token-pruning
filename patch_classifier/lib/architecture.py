import torch.nn as nn


class PatchClassifier(nn.Module):
    def __init__(self, patch_size: int = 28, n_features: int = 256):

        super(PatchClassifier, self).__init__()

        self.norm = nn.LayerNorm([3, patch_size, patch_size])
        self.conv = nn.Conv2d(3, n_features, patch_size, patch_size)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(n_features, 1)

    def forward(self, x):
        x = self.norm(x)
        x = self.conv(x)
        x = self.gelu(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.fc(x).squeeze(-1)
        return x
