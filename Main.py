import Model
import Library
import Training
import yaml
import torch.nn as nn
import torch.optim as optim
import time
import argparse

##################################################
# Code to launch the different types of attacks
# To train without any attacks:  python Main.py --attack none
# To train with only PGD attacks: python Main.py --attack pgd
# To train with all attacks: python Main.py --attack all
##################################################

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

    # Parse arguments
    parser = argparse.ArgumentParser(description="Run specified experiments")
    parser.add_argument('--attack', type=str, default='all', help='Path to configuration file')
    args = parser.parse_args()

    # Load configuration
    config = load_config()

    # Select model based on configuration
    model = get_model_from_config(config)

    # Hyperparameters and setup from config
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config['model']['learning_rate'])
    batch_size = config['model']['batch_size']
    noise_percentage = config['model']['noise_percentage']
    train_loader = Library.load_cifar('train', batch_size, noise_percentage)
    test_loader = Library.load_cifar('test', batch_size, noise_percentage)

    # Run experiments and collect results
    results, execution_times = run_experiments(config, model, criterion, optimizer, train_loader, test_loader, args.attack)

    # Save results to JSON
    Library.save_results_to_json(results)

    # Visualize results
    df_train, df_test = Library.load_and_display_results_from_json('./outputs/results.json')
    Library.formatdataframe(df_train, "./outputs/results_train.csv")
    Library.formatdataframe(df_test, "./outputs/results_test.csv")
    Library.visualize_results(df_train)

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
        'train_accuracies': trainer.train_accuracies,
        'test_accuracies': trainer.test_accuracies,
        'test_accuracies_like_train' : trainer.test_accuracies_like_train,
        'test_precisions': trainer.test_precisions,
        'test_recall': trainer.test_recalls,
        'test_f1_scores': trainer.test_f1_scores
    }

##################################################
# Function to run experiments
##################################################
def run_experiments(config, model, criterion, optimizer, train_loader, test_loader, attack_type):
    results = []
    execution_times = {}

    # Normal training
    if attack_type in ['none', 'all']:
        start_time = time.time()
        results.append(train_model(None, model, criterion, optimizer, train_loader, test_loader, config['model']['n_epochs'], 'normal'))
        end_time = time.time()
        execution_times['normal'] = end_time - start_time

    # Loop through the attack configurations
    for attack_conf in config['attacks']:

        # Check if the attack type matches the requested type or if we should run all
        if attack_conf['type'] != 'normal' and (attack_type == 'all' or attack_conf['type'] == attack_type):
            attack_class = getattr(Model, attack_conf['attack_class'])

            # Retrieve alpha, num_iter, and c for the current attack
            alpha = attack_conf.get('alpha', config['adversarial_training'].get('alpha'))
            num_iter = attack_conf.get('num_iter', config['adversarial_training'].get('num_iter'))
            c = attack_conf.get('c', None)  # Default to None if not applicable

            for eps in attack_conf['epsilons']:
                attack_name = f"{attack_conf['type']}_eps_{eps}"

                # Handle attack initialization based on the type
                if attack_conf['type'] == 'fgsm':
                    attack = attack_class(model, eps)
                elif attack_conf['type'] in ['pgd', 'pgd_l2']:
                    attack = attack_class(model, eps, alpha, num_iter)
                elif attack_conf['type'] in ['cw_l2', 'cw_linf']:
                    attack = attack_class(model, eps, alpha, num_iter, c)
                else:
                    raise ValueError(f"Type of attack not supported : {attack_conf['type']}")

                start_time = time.time()
                results.append(train_model(attack, model, criterion, optimizer, train_loader, test_loader, config['model']['n_epochs'], f"{attack_conf['type']}_eps_{eps}"))
                end_time = time.time()
                execution_times[attack_name] = end_time - start_time

    # Export execution times to JSON file
    Library.save_results_to_json(execution_times, './outputs/execution_times.json')

    return results, execution_times

if __name__ == "__main__":
    main()