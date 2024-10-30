# CIFAR-10 Classification with Adversarial Attacks

This project implements an image classification model on the CIFAR-10 dataset, with tests of adversarial attacks to evaluate the model's robustness.

## Prerequisites

- Python 3.x
- pip
- conda (optional)

## Installation

1. Clone the repository:

    ```sh
    git clone <REPOSITORY_URL>
    cd <REPOSITORY_NAME>
    ```

2. Install the dependencies:

    ```sh
    pip install -r requirements.txt
    ```

3. (Optional) Create a conda environment and install the dependencies:

    ```sh
    conda env create -f requirements.yml
    conda activate <ENVIRONMENT_NAME>
    ```

## Usage

### Loading the CIFAR-10 Dataset

The `Library.py` file contains a function to load the CIFAR-10 dataset:

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_cifar(split, batch_size, noise_percentage=0.0):
    train = True if split == 'train' else False
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x + noise_percentage * torch.randn_like(x))
    ])
    dataset = datasets.CIFAR10("./docs", train=train, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train)
```

### Training and Evaluation

The Training.py file contains the code to train the model and evaluate its accuracy, with or without adversarial attacks.

### Saving and Visualizing Results

The training and test results are saved in a JSON file (results.json). You can visualize the results using the functions in the Library.py file:

```
import pandas as pd
import matplotlib.pyplot as plt
import json

def save_results_to_json(results, file_path='results.json'):
    with open(file_path, 'w') as f:
        json.dump(results, f, indent=4)

def load_results_to_dataframe(file_path='results.json'):
    with open(file_path, 'r') as f:
        results = json.load(f)

    types = []
    train_losses = []
    test_accuracies = []

    for result in results:
        types.append(result['type'])
        train_losses.append(result['train_losses'])
        test_accuracies.append(result['test_accuracies'])

    df = pd.DataFrame({
        'Type': types,
        'Train Losses': train_losses,
        'Test Accuracies': test_accuracies
    })

    return df

def visualize_results(df):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    for i in range(len(df)):
        plt.plot(df['Train Losses'].iloc[i], label=df['Type'].iloc[i])
    plt.title('Train Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    for i in range(len(df)):
        plt.plot(df['Test Accuracies'].iloc[i], label=df['Type'].iloc[i])
    plt.title('Test Accuracies')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()
```

### Running Specific Parts of the Code

You can run specific parts of the code, such as training a model without attacks or with a specific type of attack (e.g., PGD), by using command-line arguments:  

To train without any attacks:  
```python Main.py --attack none```

To train with only PGD attacks:  
```python Main.py --attack pgd```

To train with all attacks:  
```python Main.py --attack all```
