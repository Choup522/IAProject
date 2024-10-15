import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt


##################################################
# Custom Dataset to add noise to a percentage of images
##################################################
class NoisyCIFAR10(Dataset):

    def __init__(self, dataset, noise_percentage, noise_std=0.1):
        self.dataset = dataset
        self.noise_percentage = noise_percentage
        self.noise_std = noise_std

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]

        # Decide whether to add noise to the image
        if np.random.rand() < self.noise_percentage:
            # Add Gaussian noise
            noise = torch.randn_like(image) * self.noise_std
            noisy_image = image + noise
            # Clip values to keep them in the range [0, 1]
            noisy_image = torch.clamp(noisy_image, 0, 1)
            return noisy_image, label
        else:
            # Return the clean image
            return image, label


##################################################
# Function to load CIFAR-10 dataset
##################################################
def load_cifar(split, batch_size, noise_percentage=0, noise_std=0.1):

    # Set train to True if split is 'train'
    train = True if split == 'train' else False

    # Load the CIFAR-10 dataset
    dataset = datasets.CIFAR10("./docs", train=train, download=True, transform=transforms.ToTensor())

    # If noise_percentage > 0, wrap the dataset in NoisyCIFAR10 to add noise
    if noise_percentage > 0:
        dataset = NoisyCIFAR10(dataset, noise_percentage=noise_percentage, noise_std=noise_std)

    return DataLoader(dataset, batch_size=batch_size, shuffle=train)


##################################################
# Function to save results in a JSON file
##################################################
def save_results_to_json(results, file_path='results.json'):
    with open(file_path, 'w') as f:
        json.dump(results, f, indent=4)


##################################################
# Function to display the results
##################################################
def visualize_results(df):
  # size of the figure
  plt.figure(figsize=(12, 6))

  # Graph of training losses (train losses)
  plt.subplot(1, 2, 1)
  for i in range(len(df)):
    plt.plot(df['Train Losses'].iloc[i], label=df['Type'].iloc[i])
  plt.title('Train Losses')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend()

  # graph of test accuracies (test accuracies)
  plt.subplot(1, 2, 2)
  for i in range(len(df)):
    plt.plot(df['Test Accuracies'].iloc[i], label=df['Type'].iloc[i])
  plt.title('Test Accuracies')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.legend()

  # Display the figure
  plt.tight_layout()
  plt.show()

##################################################
# Function to read the json file
##################################################
def load_and_display_results_from_json(file_path='results.json'):

  # Loading the results
  with open(file_path, 'r') as f:
    results = json.load(f)

  # Initialize lists to store the data
  types = []
  train_losses = []
  test_accuracies = []

  # Display the results
  for result in results:
    types.append(result['type'])
    train_losses.append(result['train_losses'])
    test_accuracies.append(result['test_accuracies'])

  # Create a DataFrame from the extracted data
  df = pd.DataFrame({
    'Type': types,
    'Train Losses': train_losses,
    'Test Accuracies': test_accuracies
  })

  return df