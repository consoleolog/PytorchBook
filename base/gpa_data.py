import os
import numpy as np 
import torch
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
    scores, admits = [], []
    for score, admit in batch: 
        scores.append(score)
        admits.append(admit)  
    scores = torch.tensor(np.array(scores), dtype=torch.float32)
    admits = torch.tensor(np.array(admits).reshape(-1, 1), dtype=torch.float32)
    return scores, admits    


dl = DataLoader(gpa_dataset, batch_size=8, shuffle=True, collate_fn=collater)