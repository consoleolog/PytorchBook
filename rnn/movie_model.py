import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from movie_data import dl, vocab, tokenizer

class MovieModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(MovieModel, self).__init__()
        self.embed = nn.EmbeddingBag(vocab_size, embed_dim)
        self.hidden = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
        )
        self.fc = nn.Linear(embed_dim, num_class)
    
    def forward(self, text, offsets, labels=None, train=True):
        if train:
            loss = self.train_step(text, labels, offsets)
            return loss 
        else:
            out = self.predict(text, offsets)
            return out 
    
    def train_step(self, text, labels, offsets):
        out = self.predict(text, offsets)
        loss = self.loss(out, labels)
        return loss 
    
    def predict(self, text, offsets):
        embed = self.embed(text, offsets)
        latent = self.hidden(embed)
        out = self.fc(latent)
        return out 
    
    def loss(self, y_hat, y):
        loss = F.cross_entropy(y_hat, y)
        return loss 
    
m = MovieModel(len(vocab), 64, 10)
optim = torch.optim.SGD(m.parameters(), lr=0.1)

if __name__ == "__main__":
    for e in range(3):
        for i, (label, text, offset) in enumerate(dl):
            loss = m(text, offset, label)
            optim.zero_grad()
            loss.backward()
            optim.step()
            print(f"\r{i} / {len(dl)} | loss = {loss:.3f}", end="")
            if i % 2000 == 0:
                print()
            
    review1 = "It is good and fantastic"
    review2 = "It is bad and terrible"
    review1 = torch.tensor(vocab(tokenizer(review1)))
    review2 = torch.tensor(vocab(tokenizer(review2)))
    result1 = m.predict(review1, offsets=torch.tensor([0]))
    result2 = m.predict(review2, offsets=torch.tensor([0]))
    breakpoint()