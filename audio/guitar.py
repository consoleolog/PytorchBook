from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from torch.utils.data import DataLoader, Dataset
from glob import glob
import librosa

def add_noise(data, noise_level=0.005):
    """노이즈 추가
    Args:
        noise_level (float, optional): 노이즈 강도 높을수록 많은 노이즈 추가
    """
    noise = np.random.randn(len(data))
    augmented_data = data + noise_level * noise
    augmented_data = np.clip(augmented_data, -1, 1)
    return augmented_data

class GuitarDataset(Dataset):
    def __init__(self):
        super(GuitarDataset, self).__init__()
        path = Path(__file__).parent / "Guitar Dataset"
        data_paths = glob(f"{path}/**/*.wav")
        self.data = [self.get_sample(p) for p in data_paths]

    @staticmethod
    def get_sample(path):
        waveform, sr = librosa.load(path)
        noise_waveform = add_noise(waveform)
        return waveform, noise_waveform, sr

    def __getitem__(self, idx):
        return self.data[idx]
    def __len__(self):
        return len(self.data)

class DenoiseModel(nn.Module):
    def __init__(
        self,
        sample_rate,
        n_fft,
        win_length,
        hop_length,
        n_mels,
        in_features,
    ):
        super(DenoiseModel, self).__init__()
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm="slaney",
            n_mels=n_mels,
            mel_scale="htk"
        )
        self.flatten = nn.Flatten()
        self.encoder = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, in_features),
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, in_features),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.flatten(x)
        z = self.encoder(z)
        out = self.decoder(z)
        return out

def vae_loss(x, x_hat, mean, logvar):
    recon_loss = F.binary_cross_entropy(x_hat, x)

    var = torch.exp(logvar)
    kl_loss = 0.5 * torch.mean(mean ** 2 + var - logvar - 1)
    return recon_loss, kl_loss

if __name__ == "__main__":
    dataset = GuitarDataset()
    dl = DataLoader(dataset, batch_size=32, shuffle=True)
    m = DenoiseModel(
        sample_rate=22050,
        n_fft=1024,
        win_length=512,
        hop_length=512,
        n_mels=128,
        in_features=44100,
    )
    optim = torch.optim.Adam(m.parameters(), lr=0.001, betas=(0.9, 0.999))
    for e in range(1, 20 + 1):
        for i, (wav, noise, sr) in enumerate(dl):
            z = m.encoder(wav)
            z1, z2 = z[:, :z.shape[1] // 2], z[:, z.shape[1] // 2:]
            sig = torch.exp(0.5 * z2)
            z = z1 + sig * torch.randn_like(z1)

            x_hat = m.decoder(z)

            rl, kl = vae_loss(wav, x_hat, z1, z2)
            loss = rl + kl
            loss.backward()
            optim.step()
            print(f"\rEpoch {e}: loss = {loss.item():.3f} = {rl.item():.3f} + {kl.item():.3f}", end="")
        print()
