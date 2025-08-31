import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, dim, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction),
            nn.ReLU(),
            nn.Linear(dim // reduction, dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        w = self.fc(x)
        return x * w

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        out = F.mish(self.norm1(self.fc1(x)))
        out = self.norm2(self.fc2(out))
        return F.mish(out + x)

class NNUEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(4869, 1024)
        self.norm = nn.LayerNorm(1024)

        self.res_blocks = nn.Sequential(
            ResidualBlock(1024),
            SEBlock(1024),
            ResidualBlock(1024),
            SEBlock(1024),
            ResidualBlock(1024),
        )

        self.policy_head = nn.Sequential(
            nn.Linear(1024, 256),
            nn.Mish(),
            nn.Linear(256, 4096)
        )

        self.value_head = nn.Sequential(
            nn.Linear(1024, 128),
            nn.Mish(),
            nn.Linear(128, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = F.mish(self.norm(self.input_layer(x)))
        x = self.res_blocks(x)
        policy_logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        return policy_logits, value
