import Model
import Library
import Training
import torch.nn as nn
import torch.optim as optim

# Defining the model, criterion, optimizer, and data loaders
model = Model.ConvModel()
criterion = nn.CrossEntropyLoss()                           # Loss function (for example, nn.CrossEntropyLoss())
optimizer = optim.SGD(model.parameters(), lr=.05)           # Optimizer (ofr example, optim.Adam(model.parameters(), lr=0.001))
batch_size = 100                                            # Size of the batch
train_loader = Library.load_cifar('train', batch_size) # Dataloader for training data
test_loader = Library.load_cifar('test', batch_size)   # Dataloader for test data
n_epochs = 2                                                # Number of epochs
alpha = 0.01                                                # Alpha for PGD
num_iter = 15                                               # Number of iterations for PGD

# TESTER AUSSI LE NB DE EPOCHS ET CHANGER l'OPTIMIZER ??

# List to store the results
results_list = []

# Normal training
trainer = Training.Training(model, criterion, optimizer, train_loader, test_loader, attack=None, n_epochs=n_epochs)
trainer.adversarial_train_model()
trainer.eval_model()
results_list.append({
    'type': 'normal',
    'train_losses': trainer.train_losses,
    'test_accuracies': trainer.test_accuracies
})

# Adversarial training with PGD with different epsilons
for eps in [0.02, 0.04, 0.06]:
    attack = Model.ProjectedGradientDescent(model, eps, alpha, num_iter)
    trainer = Training.Training(model, criterion, optimizer, train_loader, test_loader, attack=attack, n_epochs=n_epochs)
    trainer.adversarial_train_model()
    trainer.eval_model()

    results_list.append({
        'type': f'adversarial_pgd_eps_{eps}',
        'train_losses': trainer.train_losses,
        'test_accuracies': trainer.test_accuracies
    })

# Adversarial training with PGD l2 with different epsilons
for eps in [0.02, 0.04, 0.06]:
    attack = Model.ProjectedGradientDescent_l2(model, eps, alpha, num_iter)
    trainer = Training.Training(model, criterion, optimizer, train_loader, test_loader, attack=attack, n_epochs=n_epochs)
    trainer.adversarial_train_model()
    trainer.eval_model()

    results_list.append({
        'type': f'adversarial_pgd_l2_eps_{eps}',
        'train_losses': trainer.train_losses,
        'test_accuracies': trainer.test_accuracies
    })

# Adversarial training with FGSM with different epsilons
for eps in [0.02, 0.04, 0.06]:
    attack = Model.FastGradientSignMethod(model, eps)
    trainer = Training.Training(model, criterion, optimizer, train_loader, test_loader, attack=attack, n_epochs=n_epochs)
    trainer.adversarial_train_model()
    trainer.eval_model()

    results_list.append({
        'type': f'adversarial_fgsm_eps_{eps}',
        'train_losses': trainer.train_losses,
        'test_accuracies': trainer.test_accuracies
    })

# Display results for each configuration
for result in results_list:
    print(f"Résultats pour {result['type']} :")
    print("Losses :", result['train_losses'])
    print("Accuracy :", result['test_accuracies'])

# Creation of the dataframe
df = Library.results_to_dataframe(results_list)

# Export the dataframe to a CSV file
df.to_csv('results.csv', index=False)

# Display the graph
Library.visualize_results(df)