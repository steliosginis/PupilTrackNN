import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    """
    Residual block — learns the difference from the input
    rather than a full transformation. Better gradient flow.
    """
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))  # skip connection


class PupilNet(nn.Module):
    """
    Upgraded CNN regression network for pupil center detection.
    Takes image + Hough center hint as input.

    Input 1: 128x128 grayscale image (1 channel)
    Input 2: (cx, cy) Hough estimate normalized to 0-1
    Output:  (cx, cy) refined center normalized to 0-1
    """
    def __init__(self):
        super(PupilNet, self).__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            ResBlock(32),
            nn.MaxPool2d(2, 2),              # 128 → 64

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResBlock(64),
            nn.MaxPool2d(2, 2),              # 64 → 32

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            ResBlock(128),
            nn.MaxPool2d(2, 2),              # 32 → 16

        )

        # CNN output size: 128 * 16 * 16 = 32768
        # + 2 for Hough hint (cx, cy)
        self.regressor = nn.Sequential(
            nn.Linear(128 * 16 * 16 + 2, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 2)
        )

    def forward(self, image, hough_hint):
        """
        Args:
            image:      (batch, 1, 128, 128) tensor
            hough_hint: (batch, 2) tensor — normalized (cx, cy) from Hough
        """
        features = self.features(image)
        features = features.view(features.size(0), -1)  # flatten

        # Concatenate CNN features with Hough hint
        combined = torch.cat([features, hough_hint], dim=1)

        return self.regressor(combined)