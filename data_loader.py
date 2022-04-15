import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import math
import os

os.system('cls')

class WineDataset(Dataset):

    def __init__(self):
        
        xy = np.loadtxt('wine.csv', delimiter = ',', dtype = np.float32, skiprows = 1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]])
        self.n_samples = xy.shape[0]
        
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples