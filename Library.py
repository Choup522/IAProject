from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pandas as pd
#import matplotlib.pyplot as plt

# A TESTER
import matplotlib
matplotlib.use('TkAgg')  # Utilise TkAgg comme backend pour éviter les messages IMK
import matplotlib.pyplot as plt

# AMELIORATION
#SORTIE DU CSV POUR NE PAS AVOIR DES TABLEAUX DANS LES CELLULES
# GRAPHIQUE POUR CHAQUE MODELE : UN GRAPH ACC et UN GRAPH LOSS


##################################################
# Function to load CIFAR-10 dataset
##################################################
def load_cifar(split, batch_size):

  # Set train to True if split is 'train'
  train = True if split == 'train' else False

  # Load the CIFAR-10 dataset
  dataset = datasets.CIFAR10("./docs", train=split, download=True, transform=transforms.ToTensor())
  return DataLoader(dataset, batch_size=batch_size, shuffle=train)


##################################################
# Function to create dataframe from results list
##################################################
def results_to_dataframe(results_list):

  # Initialization of the list
  types = []
  train_losses = []
  test_accuracies = []

  # Read the list
  for result in results_list:
    types.append(result['type'])
    train_losses.append(result['train_losses'])
    test_accuracies.append(result['test_accuracies'])

  # Creation of the dataframe
  df = pd.DataFrame({
    'Type': types,
    'Train Losses': train_losses,
    'Test Accuracies': test_accuracies
  })

  return df

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