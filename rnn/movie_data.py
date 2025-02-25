import torch
import os.path as osp
from torch.utils.data import DataLoader, Dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from glob import glob
from os import getcwd

root_path = f"{getcwd()}/rnn"

class MovieDataset(Dataset):
    def __init__(self, mode="train"):
        super(MovieDataset, self).__init__()
        data_paths = glob(f"{root_path}/data/aclImdb/{mode}/neg/*.txt") + glob(f"{root_path}/data/aclImdb/{mode}/pos/*.txt")
        self.data = [self.get_sample(p) for p in data_paths]
        self.idx = 0 
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    @staticmethod
    def get_sample(path):
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read()
        basename = osp.basename(path)[:-4]
        label = int(basename.split("_")[-1])
        return label, txt
        
movie_dataset = MovieDataset("train")
tokenizer = get_tokenizer("basic_english")
def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)
vocab = build_vocab_from_iterator(yield_tokens(movie_dataset), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

def collater(batch):
    labels, texts, offsets = [], [], [0]
    for label, text in batch:
        labels.append(label -1)
        text_tensor = torch.tensor(vocab(tokenizer(text)), dtype=torch.long)
        texts.append(text_tensor)
        offsets.append(text_tensor.shape[0])
    labels = torch.tensor(labels, dtype=torch.long)
    texts = torch.cat(texts)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    return labels, texts, offsets

dl = DataLoader(movie_dataset, batch_size=8, shuffle=True, collate_fn=collater)

if __name__ == "__main__":
    for e in range(5):
        for i, (label, text, offset) in enumerate(dl):
            breakpoint()
#     pytorch: 2.2.2
# torchtext: 0.17.2