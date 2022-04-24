import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

#! device:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#! hyper parameters:
input_size = 784 #28*28
hidden_size = 100
num_classes = 10
num_epochs = 10
batch_size = 64
learning_rate = 0.001