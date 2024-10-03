import torch

##################################################
# Class to define the training
##################################################
class Training:

    def __init__(self, model, criterion, optimizer, train_loader, test_loader, attack=None, n_epochs=20):
        """Initialize the Training class with the model, loaders, and other parameters"""
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.attack = attack
        self.n_epochs = n_epochs

        # Variables to store losses and accuracy
        self.train_losses = []
        self.test_accuracies = []

        # Vérifie si MPS est disponible, sinon utilise le CPU
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
            self.model.to(self.device)
            print("Utilisation du GPU M1 avec MPS")
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.model.to(self.device)
            print("Utilisation du GPU avec CUDA")
        else:
            self.device = torch.device('cpu')
            print("Exécution sur le CPU")

    # Function to train the model
    def adversarial_train_model(self):

        for epoch in range(1, self.n_epochs + 1):

            valid_loss = 0.0

            # Training loop
            for n_batch, (imgs, labels) in enumerate(self.train_loader):

                imgs, labels = imgs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                # Adversarial or normal training
                if self.attack is None:
                    output = self.model(imgs)
                else:
                    delta = self.attack.compute(imgs, labels)
                    adv = imgs + delta
                    output = self.model(torch.clamp(adv, 0, 1))

                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()

                valid_loss += loss.item() * imgs.size(0)

            # Calculate the loss for the epoch
            avg_loss = valid_loss / len(self.train_loader.dataset)
            self.train_losses.append(avg_loss)
            print(f"Epoch: {epoch}, Loss: {avg_loss:.4f}")

    # Function to evaluate the model
    def eval_model(self, loader=None):
        """Function to evaluate your model on a specific loader"""
        if loader is None:
            loader = self.test_loader

        accuracy = 0.
        n_inputs = 0.

        # Evaluation loop
        for n_batch, (imgs, labels) in enumerate(loader):
            imgs, labels = imgs.to(self.device), labels.to(self.device)

            if self.attack is None:
                outputs = self.model(imgs)
            else:
                delta = self.attack.compute(imgs, labels)
                adv = imgs + delta
                outputs = self.model(torch.clamp(adv, 0, 1))

            _, predicted = torch.max(outputs.data, 1)
            accuracy += predicted.eq(labels.data).cpu().sum().numpy()
            n_inputs += imgs.shape[0]

        accuracy /= n_inputs
        self.test_accuracies.append(accuracy)

        if self.attack is None:
            print(f'Accuracy on test set: {accuracy:.4f}')
        else:
            print(f'Accuracy on test set with attack: {accuracy:.4f}')

