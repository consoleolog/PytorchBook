import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class GPADataset(Dataset):
    def __init__(self, path):
        super(GPADataset, self).__init__()
        self.data = pd.read_csv(path)
        self.data.dropna(inplace=True)
        X = self.data[["gre", "gpa", "rank"]].to_numpy() 
        y = self.data["admit"].to_numpy() 
        trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2, random_state=42)
        self.dataset = [(torch.from_numpy(trainX), torch.from_numpy(trainY))]
        self.val_dataset = [(torch.from_numpy(testX), torch.from_numpy(testY))]
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]

