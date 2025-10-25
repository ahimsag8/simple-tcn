import torch
import torch.nn as nn

class TemporalConvNet(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, 1, 1)
        )

    def forward(self, x):  # x: (B, in_dim, T)
        return self.net(x).squeeze(1)  # (B, T)

