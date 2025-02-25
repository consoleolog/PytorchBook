import os
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from gpa_data import dl 

root_path = f"{os.getcwd()}/base"

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
        
if __name__ == "__main__":
    m = GPAModel(3)

    optim = optim.Adam(m.parameters(), lr=0.1)
    for e in range(5):
        for i, (score, admit) in enumerate(dl):
            loss = m(score, admit)
            optim.zero_grad()
            loss.backward()
            optim.step()
            print(f"\r{i} / {len(dl)} | loss = {loss:.3f}", end="")
            if i % 2000 == 0:
                print()     
