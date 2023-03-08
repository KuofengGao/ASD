import torch.nn as nn
import torch.nn.functional as F


class LinearModel(nn.Module):
    def __init__(self, backbone, feature_dim, num_classes):
        super(LinearModel, self).__init__()
        self.backbone = backbone
        self.linear = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        feature = self.backbone(x)
        out = self.linear(feature)
        return out

    def update_encoder(self, backbone):
        self.backbone = backbone
