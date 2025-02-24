import os
import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms

root_path = f"{os.getcwd()}/cnn"

# Data : https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification
card_dataset =  torchvision.datasets.ImageFolder(
    root=f"{root_path}/data/train",
    transform=transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize( (0.5, 0.5, 0.5), (0.5, 0.5, 0.5) )
    ])
)

def collater(batch):
    images, labels = [], []
    for image, label in batch:
        images.append(image)
        labels.append(label)
    images = torch.tensor(np.array(images), dtype=torch.float32)
    labels = torch.tensor(np.array(labels).reshape(-1, 1), dtype=torch.float32)
    return images, labels

dl = DataLoader(card_dataset, batch_size=8, shuffle=True, collate_fn=collater)

class CardModel(nn.Module):
    def __init__(self):
        super(CardModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(32 * 25 * 25, 64)  
        self.fc2 = nn.Linear(64, 53)  

    def forward(self, x, y=None, train=True):
        if train:
            loss = self.train_step(x, y)
            return loss 
        else:
            out = self.predict(x)
            return out 
    
    def train_step(self, x, y):
        y_hat = self.predict(x)
        loss = self.loss(y, y_hat)
        return loss 
    
    def predict(self, x):
        a = self.conv1(x)
        z = F.relu(a)
        a = self.pool(z)
        a = self.dropout(a)
        a = torch.flatten(a, 1) 
        a = self.fc1(a)
        z = F.relu(a)
        a = self.fc2(z)
        out = F.softmax(a, 1)
        return out 
    
    def loss(self, y, y_hat):
        loss = F.cross_entropy(y_hat, y)
        return loss 

m = CardModel()

optim = optim.Adam(m.parameters(), lr=0.1)
for e in range(5):
    for i, (image, label) in enumerate(dl):
        breakpoint()