import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleNet(nn.Module):
    """
    Red neuronal simple para clasificación de imágenes
    Input: 3x64x64 (RGB images)
    Output: 2 clases (cat, fish)
    """

    def __init__(self, num_classes=2):
        super(SimpleNet, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        # Fully connected layers
        # Después de 3 pooling: 64 -> 32 -> 16 -> 8
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Conv Block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))

        # Conv Block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        # Conv Block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # Flatten
        x = x.view(-1, 128 * 8 * 8)

        # Fully connected
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


def create_model(num_classes=2):
    """Factory function para crear el modelo"""
    return SimpleNet(num_classes=num_classes)
