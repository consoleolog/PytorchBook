import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from card_data import dl

class CardModel(nn.Module):
    def __init__(self):
        super(CardModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1 ,padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(32 * 32 * 32, 64)  
        self.fc2 = nn.Linear(64, 53)  
        self.flat = nn.Flatten() 

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
    
    def predict(self, x): # x.shape (3, 64, 64)
        a = self.conv1(x) 
        a = F.relu(a) # a.shape (32, 64, 64)
        a = self.pool(a) # a.shape (32, 32, 32)
        a = self.flat(a)
        a = self.dropout(a)
        a = self.fc1 (a) 
        a = F.relu(a)
        a = self.fc2(a)
        out = F.softmax(a)
        return out 
    
    def loss(self, y, y_hat):
        loss = F.cross_entropy(y_hat, y)
        return loss 

if __name__ == "__main__":
    m = CardModel()

    optim = optim.Adam(m.parameters(), lr=0.1)
    for e in range(5):
        for i, (image, label) in enumerate(dl):
            loss = m(image, label)
            optim.zero_grad()
            loss.backward()
            optim.step()
            print(f"\r{i} / {len(dl)} | loss = {loss:.3f}", end="")
            if i % 2000 == 0:
                print()