import torch
import torch.nn as nn
import torch.nn.functional as f

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
        # Apply the convolution layers first, then add noise to the output
        x = self.pool(f.relu(self.conv1(x)))  # Convolution first
        x = x + torch.randn_like(x) * self.noise_std  # Add noise after the convolution

        x = self.pool(f.relu(self.conv2(x)))  # Second convolution
        x = x + torch.randn_like(x) * self.noise_std  # Add noise after the second convolution

        x = x.view(-1, 100 * 8 * 8)
        x = f.relu(self.fc1(x))
        x = x + torch.randn_like(x) * self.noise_std  # Add noise after the fully connected layer

        x = f.relu(self.fc2(x))
        x = x + torch.randn_like(x) * self.noise_std  # Add noise after the second fully connected layer

        x = self.fc3(x)  # Output layer, no noise added here
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
    x, y = x.to(device), y.to(device)
    delta = torch.zeros_like(x, requires_grad=True)

    for _ in range(self.num_iter):
        adv = x + delta
        output = self.model(adv)
        loss = self.criterion(output, y)
        loss.backward()
        delta.data = torch.clamp(delta + self.alpha * torch.sign(delta.grad.detach()), -self.eps, self.eps)
        delta.grad.zero_()

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
        adv = x + delta
        output = self.model(adv)
        loss = self.criterion(output, y)
        loss.backward()
        delta.data = torch.renorm(delta +self.alpha * torch.sign(delta.grad.detach()),2,0, self.eps)

    return delta.detach()

##################################################
# Class to define CarliniWagnerL2
##################################################
class CarliniWagnerL2:

    def __init__(self, model, eps, alpha, num_iter, c=0.1):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.num_iter = num_iter
        self.c = c
        self.criterion = nn.CrossEntropyLoss()

    def compute(self, x, y):
        # Construct CarliniWagnerL2 adversarial perturbation on the examples x
        delta = torch.zeros_like(x, requires_grad=True)

        for _ in range(self.num_iter):
            delta.requires_grad = True
            output = self.model(x + delta)

            #  Calculating the modified loss to integrate the regularization factor c
            classification_loss = self.criterion(output, y)
            perturbation_loss = torch.norm(delta.view(delta.size(0), -1), p=2, dim=1).mean() # L2 norm of the perturbation
            loss = classification_loss + self.c * perturbation_loss

            loss.backward()
            grad = delta.grad.detach()
            grad_norm = torch.norm(grad.view(grad.size(0), -1), p=2, dim=1).view(-1, 1, 1, 1)
            delta.data += self.alpha * grad / grad_norm
            delta.data = torch.min(torch.max(delta.detach(), -x), 1 - x) # Projection onto the l2 ball
            delta.data = torch.clamp(delta.detach(), -self.eps, self.eps) # Projection onto the l2 ball
            delta.grad.zero_()

        return delta.detach()

##################################################
# Class to define CarliniWagner L infini
##################################################
class CarliniWagnerLinfinity:

    def __init__(self, model, eps, alpha, num_iter, c=0.1):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.num_iter = num_iter
        self.c = c
        self.criterion = nn.CrossEntropyLoss()

    def compute(self, x, y):
        # Construct CarliniWagnerL2 adversarial perturbation on the examples x
        delta = torch.zeros_like(x, requires_grad=True)

        for _ in range(self.num_iter):
            self.model.zero_grad()
            delta.requires_grad = True
            output = self.model(x + delta)

            # Calculating the modified loss to integrate the regularization factor c
            classification_loss = self.criterion(output, y)
            perturbation_loss = torch.norm(delta.view(delta.size(0), -1), p=float('inf')).mean() # infinity norm of the perturbation
            loss = classification_loss + self.c * perturbation_loss

            loss.backward()
            grad = delta.grad.detach()

            delta.data += self.alpha * grad.sign()
            delta.data = torch.clamp(delta, -self.eps, self.eps)
            delta.data = torch.clamp(x + delta, 0, 1) - x
            delta.grad.zero_()

        return delta.detach()