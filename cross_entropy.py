import torch
import torch.nn as nn
import numpy as np
import os

os.system('cls')

def cross_entropy_loss(actual, predicted):
    loss = -np.sum(actual * np.log(predicted))
    return loss


y = np.array([1, 0, 0])
y_pred_good = np.array([0.7, 0.2, 0.1])
y_pred_bad = np.array([0.1, 0.3, 0.6])
l1 = cross_entropy_loss(y, y_pred_good)
l2 = cross_entropy_loss(y, y_pred_bad)

print(l1, l2)