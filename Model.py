import torch
import torch.nn as nn
import torch.nn.functional as F

# Gestion du CPU sous MPS ou Cuda
device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

##################################################
# Class to define the model
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
# Class to define FGSM
##################################################
class FastGradientSignMethod:

  def __init__(self, model, eps):
    self.model = model
    self.eps = eps
    self.criterion = nn.CrossEntropyLoss()

  def compute(self, x, y):
    """ Construct FGSM adversarial perturbation for examples x"""

    x, y = x.to(device), y.to(device)
    perturbation = torch.zeros_like(x, requires_grad=True)

    adv = x + perturbation
    output = self.model(adv)

    loss = self.criterion(output, y)
    loss.backward()
    delta = self.eps * torch.sign(perturbation.grad.detach())
    delta = torch.clamp(delta, -self.eps, self.eps) #en norme infini on veut pas que les perturbation soit au dessus de epsilon, dans le carré on revient dans l'espace quand on en sort

    return delta


##################################################
# Class to define PGD
##################################################
class ProjectedGradientDescent:

  def __init__(self, model, eps, alpha, num_iter):
    self.model = model
    self.eps = eps
    self.alpha = alpha
    self.num_iter = num_iter
    self.criterion = nn.CrossEntropyLoss()

  def compute(self, x, y):
    # Construct PGD adversarial perturbation on the examples x
    delta = torch.zeros_like(x, requires_grad=True)
    #delta.data = delta.data + torch.randn_like(delta) * 0.001

    for _ in range(self.num_iter):
      # The following line caused the error. It has been replaced with the correct function call.
      loss = self.criterion(self.model(x + delta), y)
      loss.backward()
      delta.data = torch.clamp(delta + self.alpha * torch.sign(delta.grad.detach()), -self.eps, self.eps)

    return delta.data

##################################################
# Class to define PGD norm l2
##################################################
class ProjectedGradientDescent_l2:

  def __init__(self, model, eps, alpha, num_iter):
    self.model = model
    self.eps = eps
    self.alpha = alpha
    self.num_iter = num_iter
    self.criterion = nn.CrossEntropyLoss()

  def compute(self, x, y):
    # Construct PGD adversarial perturbation on the examples x
    delta = torch.zeros_like(x, requires_grad=True)

    for _ in range(self.num_iter):
      loss = self.criterion(self.model(torch.clamp(x + delta, 0, 1)), y)
      loss.backward()
      delta.data = delta + self.alpha * delta.grad.detach()
      mask = (torch.norm(delta.data, p=2, dim=(1, 2, 3)) <= self.eps).float()
      delta.data = delta.data * mask.view(-1, 1, 1, 1)

      delta.data += (1 - mask.view(-1, 1, 1, 1)) * delta.data / torch.norm(delta.data, p=2, dim=(1, 2, 3)).view(-1, 1, 1, 1) * self.eps
      delta.grad.zero_()

    return delta.detach()