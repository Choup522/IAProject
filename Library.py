from torchvision import datasets, transforms
from torch.utils.data import DataLoader

##################################################
# Function to load CIFAR-10 dataset
##################################################
def load_cifar(split, batch_size):

  # Set train to True if split is 'train'
  train = True if split == 'train' else False

  # Load the CIFAR-10 dataset
  dataset = datasets.CIFAR10("./docs", train=split, download=True, transform=transforms.ToTensor())
  return DataLoader(dataset, batch_size=batch_size, shuffle=train)