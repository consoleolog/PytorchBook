import os 
import torch 
import numpy as np 
from torch.utils.data import DataLoader
from data import GPADataset
from model import GPAModel

root_path = f"{os.getcwd()}/base"

def collater(batch):
    scores, admits = [], []
    for score, admit in batch: 
        scores.append(score)
        admits.append(admit)  
    scores = torch.tensor(np.array(scores), dtype=torch.float32)
    admits = torch.tensor(np.array(admits).reshape(-1, 1), dtype=torch.float32)
    return scores, admits  

class GPATrainer:
    def __init__(self):
        self.m = GPAModel(3)
        dataset = GPADataset(f"{root_path}/data/gpascore.csv")
        self.dl = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collater)
        self.val_dl = DataLoader(dataset.val_dataset, batch_size=8, shuffle=True)
        self.optim = torch.optim.Adam(self.m.parameters(), lr=0.01)
        self.criter = torch.nn.CrossEntropyLoss()

    
    def train(self, epochs):
        for e in range(epochs):
            for i, (score, admit) in enumerate(self.dl, 1):
                out = self.m(score)
                self.optim.zero_grad()
                loss = self.criter(out, admit)
                loss.backward()
                self.optim.step()
                if i % 100 == 0:
                    print(f"\rEpoch {e}, Loss {loss.item()}", end='')
        print()
    
    def test(self):
        correct = 0 
        total = 0 
        with torch.no_grad():
            for score, admit in self.val_dl:
                out = self.m(score)
                _, predicted = torch.max(out.data, 1)

def main():
    t = GPATrainer()
    t.train(3)

if __name__ == "__main__":
    main()