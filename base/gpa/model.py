import torch.nn as nn  
import torch.nn.functional as F

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
    
    def forward(self, x):
        a = self.hidden(x)
        out = F.sigmoid(a)
        return out 
    