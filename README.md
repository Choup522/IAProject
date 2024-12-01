# CIFAR-10 Classification with Adversarial Attacks

This project implements an image classification model on the CIFAR-10 dataset, with tests of adversarial attacks to evaluate the model's robustness.

## Prerequisites

- Python 3.x
- pip
- conda (optional)

## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/Choup522/IAProject.git
    cd IAProject
    ```

2. Install the dependencies:

    ```sh
    pip install -r requirements.txt
    ```

3. (Optional) Create a conda environment and install the dependencies:

    ```sh
   conda env create -f requirements.yml
   conda activate IAProjectEnv
    ```

## Usage

### Loading the CIFAR-10 Dataset

The `Library.py` file contains a function to load the CIFAR-10 dataset:

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch

def load_cifar(split, batch_size, noise_percentage=0.0):
    train = True if split == 'train' else False
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x + noise_percentage * torch.randn_like(x))
    ])
    dataset = datasets.CIFAR10("./docs", train=train, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train)
```

The **noise_percentage** parameter can be used to add Gaussian noise to the inputs before using them to train the model.

### Description of Models

This project includes three types of models:

1. **Classic Model**: A standard convolutional neural network (CNN) for image classification.
2. **Lipschitz Model**: A CNN with Lipschitz continuity constraints to improve robustness.
3. **Randomized Model**: A CNN with added Gaussian noise to the inputs for robustness against adversarial attacks.

The `Models.py` file contains the code to create these models.

### Description of Attacks

This project includes three types of adversarial attacks:

1. **FGSM**: Fast Gradient Sign Method
2. **PGD**: Projected Gradient Descent
3. **CW**: Carlini-Wagner

### Training and Evaluation

The Training.py file contains the code to train the model and evaluate its accuracy, with or without adversarial attacks.

### Saving and Visualizing Results

The training and test results are saved in a JSON file (results.json).

### Running Specific Parts of the Code

You can run specific parts of the code, such as training a model without attacks or with a specific type of attack (e.g., PGD), by using command-line arguments:  

To train without any attacks:  
```python Main.py --attack none```

To train with only PGD attacks:  
```python Main.py --attack pgd```

To train with all attacks:  
```python Main.py --attack all```

## Results

You can view the detailed results and visualizations in the `DisplayResults.ipynb` notebook.

[View DisplayResults.ipynb](DisplayResults.ipynb)

[View ProjetIA_Adversarial.pptx](ProjetIA_Adversarial.pptx)


