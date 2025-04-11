import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

IMAGE_MEAN = (0.4914, 0.4822, 0.4465)
IMAGE_STD = (0.2023, 0.1994, 0.2010)

class PatchGenerator:
    def __init__(self, patch_size: int):
        self.patch_size = patch_size

    def __call__(self, img):
        num_channels = img.size(0)
        patches = (
            img
            .unfold(1, self.patch_size, self.patch_size) # shape (C, P, N, P)
            .unfold(2, self.patch_size, self.patch_size) # shape (C, P, P, P, P)
            .reshape(num_channels, -1, self.patch_size, self.patch_size) # shape (C, N, P, P)
        ) # shape (C, N, P, P)
        patches = patches.permute(1, 0, 2, 3) # shape (N, C, P, P)
        num_patch = patches.size(0) # N 을 가져옴
        return patches.reshape(num_patch, -1) # shape (N, C*P^2) X_p 를 의미

class PatchEmbedding(nn.Module):
    def __init__(
        self,
        patch_size: int,
        img_size: int,
        embed_dim: int
    ):
        super().__init__()
        self.patch_size = patch_size

        self.N = (img_size ** 2) // (patch_size ** 2)

        self.embedding = nn.Embedding()

    def forward(self, x: torch.Tensor):
        # x.shape (B, C, H, W)
        channel = x.shape[1]



        # x.shape (N, P^2 * C)
        pass


def make_weights(labels, n_classes):
    labels = np.array(labels)
    weight_arr = np.zeros_like(labels)
    _, counts = np.unique(labels, return_counts=True)

    for cls in range(n_classes):
        weight_arr = np.where(labels == cls, 1 / counts[cls], weight_arr)

    return weight_arr

class Flattened2DPatches:
    def __init__(
            self,
            patch_size:int = 16,
            data_name:str = "imagenet",
            img_size:int = 256,
            batch_size:int = 64
    ):
        self.patch_size = patch_size
        self.data_name = data_name
        self.img_size = img_size
        self.batch_size = batch_size

    def patch_data(self):
        train_transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.RandomCrop(self.img_size, padding=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
            PatchGenerator(self.patch_size)
        ])
        test_transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
            PatchGenerator(self.patch_size)
        ])

        if self.data_name == "cifar10":
            train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
            test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=False, transform=test_transform)
            evens = list(range(0, len(test_set), 2))
            odds = list(range(1, len(test_set), 2))

            val_set = torch.utils.data.Subset(train_set, evens)
            test_set = torch.utils.data.Subset(train_set, odds)
        else:
            raise ValueError("data_name is not allowed : {}".format(self.data_name))

        weights = make_weights(train_set.targets, len(train_set.classes))  # 가중치 계산
        weights = torch.DoubleTensor(weights)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
        train_loader = DataLoader(train_set, batch_size=self.batch_size, sampler=sampler)
        val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

def imshow(img):
    plt.figure(figsize=(100,100))
    plt.imshow(img.permute(1,2,0).numpy())
    plt.savefig("patch_example.png")

if __name__ == "__main__":
    print("Testing Flattened 2D patches..")
    BATCH_SIZE = 64
    PATCH_SIZE = 8
    IMG_SIZE = 32
    num_patches = int((IMG_SIZE * IMG_SIZE) / (PATCH_SIZE * PATCH_SIZE)) # N = ( H * W ) / P^2
    d = Flattened2DPatches(
        data_name="cifar10",
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        batch_size=PATCH_SIZE,
    )
    tl, _, _ = d.patch_data()
    images_, labels_ = next(iter(tl))
    print(images_.size(), labels_.size())

    sample = images_.reshape(BATCH_SIZE, num_patches, -1, PATCH_SIZE, PATCH_SIZE)[0]
    print("Sample image size: ", sample.size())
    imshow(torchvision.utils.make_grid(sample, nrow=int(IMG_SIZE / PATCH_SIZE)))