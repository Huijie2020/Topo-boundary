import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

class projection(nn.Module):
    def __init__(self, in_dim=12, out_dim=12):
        super().__init__()
        hidden_dim = in_dim
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=1, stride=1),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Conv2d(hidden_dim, out_dim, kernel_size=1, stride=1)
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x
