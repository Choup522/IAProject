import Model
import Library
import Training
import yaml
import torch.nn as nn
import torch.optim as optim
import time

##################################################
# Load configuration file
##################################################
def load_config(file_path='config.yaml'):
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

##################################################
# Function to select the model based on config
##################################################
def get_model_from_config(config):
    model_type = config['model']['type']

    if model_type == 'classic':
        return Model.ConvModel()  # Classical model
    elif model_type == 'lipschitz':
        lipschitz_constant = config['model'].get('lipschitz_constant', 1.0)
        return Model.LipschitzConvModel(lipschitz_constant)  # Lipschitz model
    elif model_type == 'randomized':
        noise_std = config['model'].get('noise_std', 0.1)
        return Model.RandomizedConvModel(noise_std)  # Randomised model
    else:
        raise ValueError(f"Model type '{model_type}' not recognized!")

##################################################
# Main function
##################################################
def main():

    # Load configuration
    config = load_config()

    # Select model based on configuration
    model = get_model_from_config(config)

    # Hyperparameters and setup from config
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config['model']['learning_rate'])
    batch_size = config['model']['batch_size']
    train_loader = Library.load_cifar('train', batch_size)
    test_loader = Library.load_cifar('test', batch_size)

    # Run experiments and collect results
    results, execution_times = run_experiments(config, model, criterion, optimizer, train_loader, test_loader)

    # Save results to JSON
    Library.save_results_to_json(results)

    # Create a dataframe and save as CSV
    #df = Library.results_to_dataframe(results)
    #df.to_csv('results.csv', index=False)

    # Visualize results
    df = Library.load_and_display_results_from_json('results.json')
    Library.visualize_results(df)

##################################################
# Function to train models with a specific attack
##################################################
def train_model(attack, model, criterion, optimizer, train_loader, test_loader, n_epochs, attack_type):

    trainer = Training.Training(model, criterion, optimizer, train_loader, test_loader, attack=attack, n_epochs=n_epochs)
    trainer.adversarial_train_model()
    trainer.eval_model()

    return {
        'type': attack_type,
        'train_losses': trainer.train_losses,
        'test_accuracies': trainer.test_accuracies
    }

##################################################
# Function to run experiments
##################################################
def run_experiments(config, model, criterion, optimizer, train_loader, test_loader):
    results = []
    execution_times = {}

    # Normal training
    if 'normal' in [attack['type'] for attack in config['attacks']]:
        start_time = time.time()
        results.append(train_model(None, model, criterion, optimizer, train_loader, test_loader, config['model']['n_epochs'], 'normal'))
        end_time = time.time()
        execution_times['normal'] = end_time - start_time

    # Loop through the attack configurations
    for attack_conf in config['attacks']:
        if attack_conf['type'] != 'normal':
            attack_class = getattr(Model, attack_conf['attack_class'])
            for eps in attack_conf['epsilons']:
                attack_name = f"{attack_conf['type']}_eps_{eps}"
                # Handle attack initialization based on the type
                if attack_conf['type'] == 'fgsm':
                    attack = attack_class(model, eps)  # FGSM doesn't need alpha or num_iter
                else:
                    attack = attack_class(model, eps, config['adversarial_training']['alpha'], config['adversarial_training']['num_iter'])

                start_time = time.time()
                results.append(train_model(attack, model, criterion, optimizer, train_loader, test_loader, config['model']['n_epochs'], f"{attack_conf['type']}_eps_{eps}"))
                end_time = time.time()
                execution_times[attack_name] = end_time - start_time

    # Export execution times to JSON file
    Library.save_results_to_json(execution_times, 'execution_times.json')

    return results, execution_times

if __name__ == "__main__":
    main()