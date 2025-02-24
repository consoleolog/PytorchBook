import os
import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader

root_path = f"{os.getcwd()}/base"

class GPADataset(Dataset):
    def __init__(self, path):
        super(GPADataset, self).__init__()
        self.data = pd.read_csv(path)
        self.data.dropna(inplace=True)
        self.x = self.data[["gre", "gpa", "rank"]].to_numpy() 
        self.y = self.data["admit"].to_numpy() 
    
    def __getitem__(self, idx):
        return torch.tensor(self.x[idx]), torch.tensor(self.y[idx]) 
    
    def __len__(self):
        return len(self.data)

gpa_dataset = GPADataset(f"{root_path}/data/gpascore.csv")

def collater(batch):
    X, y = [], []
    for scores, admit in batch: 
        X.append(scores)
        y.append(admit)  
    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y).reshape(-1, 1), dtype=torch.float32)
    return X, y    


dl = DataLoader(gpa_dataset, batch_size=8, shuffle=True, collate_fn=collater)

class GPAModel(nn.Module):
    def __init__(self, input_dim):
        super(GPAModel, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
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
        a = self.hidden(x)
        out = F.sigmoid(a)
        return out
    
    def loss(self, y, y_hat):
        loss = F.binary_cross_entropy(y_hat, y)
        return loss 
        
m = GPAModel(3)

optim = optim.Adam(m.parameters(), lr=0.1)
for e in range(5):
    for i, (scores, admit) in enumerate(dl):
        loss = m(scores, admit)
        optim.zero_grad()
        loss.backward()
        optim.step()
        print(f"\r{i} / {len(dl)} | loss = {loss:.3f}", end="")
        if i % 2000 == 0:
            print()     
