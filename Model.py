import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.nn.utils as utils

# Gestion du CPU sous MPS ou Cuda
device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

##################################################
# Class to define the classical model
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
    x = self.pool(f.relu(self.conv1(x)))
    x = self.pool(f.relu(self.conv2(x)))
    x = x.view(-1, 100 * 8 * 8)
    x = f.relu(self.fc1(x))
    x = f.relu(self.fc2(x))
    x = self.fc3(x)
    return x

##################################################
# Class to define a Lipschitz constrained ConvModel
##################################################
class LipschitzConvModel(ConvModel):

    def __init__(self, lipschitz_constant=1.0):
        super(LipschitzConvModel, self).__init__()
        self.lipschitz_constant = lipschitz_constant  # Lipschitz parameter

    def forward(self, x):
        # Normalization of the weights for each layer
        self.conv1.weight.data = utils.clip_grad_norm_(self.conv1.weight, max_norm=self.lipschitz_constant)
        self.conv2.weight.data = utils.clip_grad_norm_(self.conv2.weight, max_norm=self.lipschitz_constant)
        self.fc1.weight.data = utils.clip_grad_norm_(self.fc1.weight, max_norm=self.lipschitz_constant)
        self.fc2.weight.data = utils.clip_grad_norm_(self.fc2.weight, max_norm=self.lipschitz_constant)
        self.fc3.weight.data = utils.clip_grad_norm_(self.fc3.weight, max_norm=self.lipschitz_constant)

        # Forward pass normal
        return super(LipschitzConvModel, self).forward(x)

##################################################
# Class to define a Randomized ConvModel
##################################################
class RandomizedConvModel(ConvModel):

    def __init__(self, noise_std=0.1):
        super(RandomizedConvModel, self).__init__()
        self.noise_std = noise_std  # Écart type du bruit

    def forward(self, x):
        # Adding Gaussian noise to the activations
        x = self.pool(f.relu(self.conv1(x) + torch.randn_like(x) * self.noise_std))
        x = self.pool(f.relu(self.conv2(x) + torch.randn_like(x) * self.noise_std))
        x = x.view(-1, 100 * 8 * 8)
        x = f.relu(self.fc1(x) + torch.randn_like(x) * self.noise_std)
        x = f.relu(self.fc2(x) + torch.randn_like(x) * self.noise_std)
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
    delta = torch.clamp(delta, -self.eps, self.eps) # With infinity norm we don't want the perturbation to be above epsilon, in the square we come back in the space when we leave it

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

    for _ in range(self.num_iter):
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

##################################################
# Class to define CarliniWagnerL2
##################################################
class CarliniWagnerL2:

    def __init__(self, model, eps, alpha, num_iter):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.num_iter = num_iter
        self.criterion = nn.CrossEntropyLoss()

    def compute(self, x, y):
        # Construct CarliniWagnerL2 adversarial perturbation on the examples x
        delta = torch.zeros_like(x, requires_grad=True)

        for _ in range(self.num_iter):
            delta.requires_grad = True
            output = self.model(x + delta)
            loss = self.criterion(output, y)
            loss.backward()

            grad = delta.grad.detach()
            grad_norm = torch.norm(grad.view(grad.size(0), -1), p=2, dim=1).view(-1, 1, 1, 1)
            delta.data += self.alpha * grad / grad_norm
            delta.data = torch.min(torch.max(delta.detach(), -x), 1 - x) # Projection onto the l2 ball
            delta.data = torch.clamp(delta.detach(), -self.eps, self.eps) # Projection onto the l2 ball
            delta.grad.zero_()

        return delta.detach()