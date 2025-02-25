import os
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

root_path = f"{os.getcwd()}/cnn"

# Data : https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification
card_dataset =  torchvision.datasets.ImageFolder(
    root=f"{root_path}/data/train",
    transform=transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    ])
)

def collater(batch):
    images, labels = [], []
    for image, label in batch:
        images.append(image)
        labels.append(label)
    images = torch.tensor(np.array(images), dtype=torch.float32) # 8, 3, 64, 64
    labels = torch.tensor(labels)  # (8,)
    labels = F.one_hot(labels, num_classes=len(card_dataset.classes)).float()  # (8, 53)
    return images, labels

dl = DataLoader(card_dataset, batch_size=8, shuffle=True, collate_fn=collater)

if __name__ == "__main__":
    for e in range(10):
        for i, (image, label) in enumerate(dl):
            conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1 ,padding=1)
            # nn.MaxPool2d()
            breakpoint()