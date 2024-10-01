import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import torch.nn.functional as F

cuda = torch.cuda.is_available()

##################################################
# Classe pour définir le modèle
##################################################
class ConvModel(torch.nn.Module):

  def __init__(self): # Define the layers of the network
    super(ConvModel, self).__init__()
    self.conv1 = nn.Conv2d(3, 6, 3, padding=1) # 3 input channels, 6 output channels, 3x3 kernel
    self.pool = nn.MaxPool2d(2) # 2x2 kernel
    self.conv2 = nn.Conv2d(6, 100, 3, padding=1) # 6 input channels, 16 output channels, 5x5 kernel
    self.fc1 = nn.Linear(100 * 8 * 8, 120) # 16x5x5 input features, 120 output features
    self.fc2 = nn.Linear(120, 84) # 120 input features, 84 output features
    self.fc3 = nn.Linear(84, 10) # 84 input features, 10 output features

  def forward(self, x): # Define the forward pass
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = x.view(-1, 100 * 8 * 8)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

##################################################
# Classe pour définir le FGSM
##################################################
class FastGradientSignMethod:

  def __init__(self, model, eps):
    self.model = model
    self.eps = eps
    self.criterion = nn.CrossEntropyLoss()

  def compute(self, x, y):
    """ Construct FGSM adversarial perturbation for examples x"""
    if cuda:
      x, y = x.cuda(), y.cuda()
    perturbation = torch.zeros_like(x, requires_grad=True)

    adv = x + perturbation
    output = self.model(adv)

    loss = self.criterion(output, y)
    loss.backward()
    delta = self.eps * torch.sign(perturbation.grad.detach())
    delta = torch.clamp(delta, -self.eps, self.eps) #en norme infini on veut pas que les perturbation soit au dessus de epsilon, dans le carré on revient dans l'espace quand on en sort

    return delta


##################################################
# Classe pour définir le PGD
##################################################
class ProjectedGradientDescent:

  def __init__(self, model, eps, alpha, num_iter):
    self.model = model
    self.eps = eps
    self.alpha = alpha
    self.num_iter = num_iter
    self.criterion = nn.CrossEntropyLoss()

  def compute(self, x, y):
    """ Construct PGD adversarial pertubration on the examples x."""
    delta = torch.zeros_like(x, requires_grad=True)
    #delta.data = delta.data + torch.randn_like(delta) * 0.001

    for _ in range(self.num_iter):
      # The following line caused the error. It has been replaced with the correct function call.
      loss = criterion(self.model(x + delta), y)
      loss.backward()
      delta.data = torch.clamp(delta + self.alpha * torch.sign(delta.grad.detach()), -self.eps, self.eps)

    return delta.data