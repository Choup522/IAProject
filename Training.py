import torch
import Model
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

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

        # Variables to store losses and accuracy on the training set
        self.train_losses = []
        self.train_accuracies = []

        # Variables to store metrics on the test set
        self.test_accuracies = []
        self.test_precisions = []
        self.test_recalls = []
        self.test_f1_scores = []

        # check if MPS is available, otherwise use CPU
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
            self.model.to(self.device)
            print("GPU M with MPS")
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.model.to(self.device)
            print("GPU with CUDA")
        else:
            self.device = torch.device('cpu')
            print("CPU, Good luck!")

    # Function to train the model
    def adversarial_train_model(self):

        for epoch in range(1, self.n_epochs + 1):

            valid_loss = 0.0
            total_correct = 0.0  # Cumulative number of correct predictions
            n_inputs = 0.0  # Cumulative number of inputs

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

                # Clip gradients only for the Lipschitz model
                if isinstance(self.model, Model.LipschitzConvModel):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.model.lipschitz_constant)

                self.optimizer.step()

                valid_loss += loss.item() * imgs.size(0)

                # Calculate accuracy during training
                _, predicted = torch.max(output.data, 1)
                total_correct += predicted.eq(labels.data).sum().item()
                n_inputs += imgs.size(0)

            # Calculate the loss for the epoch
            avg_loss = valid_loss / len(self.train_loader.dataset)
            accuracy = total_correct / n_inputs
            self.train_losses.append(avg_loss)
            self.train_accuracies.append(accuracy)
            print(f"Epoch: {epoch}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

    # Function to evaluate the model
    def eval_model(self, loader=None):
        """Function to evaluate your model on a specific loader"""
        if loader is None:
            loader = self.test_loader

        all_labels = []
        all_predictions = []

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

            # Collect predictions and true labels for metrics
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # Calculate accuracy, precision, recall, and F1 score using scikit-learn
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='weighted',  zero_division=0)
        recall = recall_score(all_labels, all_predictions, average='weighted')
        f1 = f1_score(all_labels, all_predictions, average='weighted')

        # Append the accuracy to the accuracy list
        self.test_accuracies.append(accuracy)
        self.test_recalls.append(recall)
        self.test_precisions.append(precision)
        self.test_f1_scores.append(f1)

        if self.attack is None:
            print(f'Accuracy on test set: {accuracy:.4f}')
        else:
            print(f'Accuracy on test set with attack: {accuracy:.4f}')

        print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
