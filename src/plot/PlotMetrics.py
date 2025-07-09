import argparse
import glob
import os

from src.data.DatasetConstants import SPLIT, NB_CLIENTS, BATCH_SIZE, STEP_SIZE, MOMENTUM, SCHEDULER_PARAMS, \
    RUNNING_CLIENTS
from src.utils.LoggingWriter import LoggingWriter
from src.utils.PlotUtilities import plot_values, plot_weights
from src.utils.Utilities import get_project_root

def extract_number(chaine):
    # Diviser la chaîne par '_' et prendre le dernier élément avant l'extension
    dernier_partie = chaine.split('_')[2]
    # Enlever l'extension .pkl et convertir en entier
    nombre = int(dernier_partie)
    return nombre

def my_dict(all_algos, all_seeds):
    return {algo: {s: [] for s in all_seeds} for algo in all_algos}


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Name of the dataset.",
        required=True,
    )
    parser.add_argument(
        "--inner_iterations",
        type=int,
        help="Number of inner iterations (if not provided, defaults is None leading to take the dataset's size).",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--batch_size_alignement",
        type=int,
        help="Batch size for gradient alignement is weights computations.",
        required=False,
        default=512
    )
    args = parser.parse_args()
    dataset_name = args.dataset_name
    inner_iterations = args.inner_iterations
    batch_size_alignement = args.batch_size_alignement
    folder = "final"

    assert dataset_name in ["exam_llm", "mnist", "mnist_iid", "cifar10", "cifar10_iid", "heart_disease", "tcga_brca", "ixi", "liquid_asset",
                            "synth", "synth_complex"], "Dataset not recognized."
    print(f"### ================== DATASET: {dataset_name} ================== ###")

    nb_initial_epochs = 0

    if dataset_name in ["heart_disease", "ixi"]:
        all_algos = ["All-for-one-bin", "All-for-one-cont", "Local", "FedAvg", "Ditto", "Cobo", "Wga-bc", "Apfl"]
    else:
        all_algos = ["All-for-one-bin", "All-for-one-cont", "All-for-one-opt", "Local", "FedAvg", "Ditto", "Cobo",
                     "Wga-bc", "Apfl"]
    all_seeds = [127, 496, 1729]  # Mersenne number, Perfect number, Ramanujan number

    train_epochs, train_losses, train_accuracies = my_dict(all_algos, all_seeds), my_dict(all_algos, all_seeds), my_dict(all_algos, all_seeds)
    test_epochs, test_losses, test_accuracies = my_dict(all_algos, all_seeds), my_dict(all_algos, all_seeds), my_dict(all_algos, all_seeds)
    weights, ratio = my_dict(all_algos, all_seeds), my_dict(all_algos, all_seeds)

    for algo_name in all_algos:
        assert algo_name in ["All-for-one-bin", "All-for-one-cont", "All-for-one-opt", "Local", "FedAvg",
                             "Ditto", "Cobo", "Wga-bc", "Apfl"], \
            "Algorithm not recognized."
        print(f"--- ================== ALGO: {algo_name} ================== ---")

        for seed in all_seeds:

            root = get_project_root()
            pickle_folder = f'{root}/pickle/{folder}/{dataset_name}/{algo_name}/{seed}'

            # Use glob to find all files matching the pattern
            if dataset_name in ["mnist", "cifar10"]:
                split_type = SPLIT[dataset_name]
                regex = f'logging_writer_*_N{NB_CLIENTS[dataset_name]}_'\
                        f'b{BATCH_SIZE[dataset_name]}_LR{STEP_SIZE[dataset_name]}_' \
                        f's{SCHEDULER_PARAMS[dataset_name][0]}_m{MOMENTUM[dataset_name]}_' \
                        f'inner{inner_iterations}_bAl{batch_size_alignement}_'\
                        f'{split_type}.pkl'
                file_pattern = os.path.join(pickle_folder, regex)
            else:
                regex = f'logging_writer_*_N{NB_CLIENTS[dataset_name]}_'\
                        f'b{BATCH_SIZE[dataset_name]}_LR{STEP_SIZE[dataset_name]}_'\
                        f's{SCHEDULER_PARAMS[dataset_name][0]}_m{MOMENTUM[dataset_name]}_'\
                        f'inner{inner_iterations}_bAl{batch_size_alignement}.pkl'
                file_pattern = os.path.join(pickle_folder, regex)
            matching_files = glob.glob(file_pattern)

            # Extract the file names from the full paths
            file_names = sorted([os.path.basename(file) for file in matching_files
                                 if os.path.basename(file) != "logging_writer_central.pkl"], key=extract_number)
            if len(file_names) == 0:
                raise ValueError(f"There is no corresponding files in {pickle_folder} for:\n {regex}")
            for name in file_names[:RUNNING_CLIENTS]:

                writer = LoggingWriter.load(pickle_folder, name)

                train_epochs[algo_name][seed].append(writer.retrieve_information("train_accuracy")[0])
                train_accuracies[algo_name][seed].append(writer.retrieve_information("train_accuracy")[1])
                train_losses[algo_name][seed].append(writer.retrieve_information("train_loss")[1])

                test_epochs[algo_name][seed].append(writer.retrieve_information("test_accuracy")[0])
                test_accuracies[algo_name][seed].append(writer.retrieve_information("test_accuracy")[1])
                test_losses[algo_name][seed].append(writer.retrieve_information("test_loss")[1])

                weights[algo_name][seed].append(writer.retrieve_histogram_information("weights")[1])
                ratio[algo_name][seed].append(writer.retrieve_histogram_information("ratio")[1])

            if algo_name not in ["FedAvg", "FedNova", "Wga-bc", "Apfl"]:
                plot_weights(weights[algo_name][all_seeds[0]], dataset_name, algo_name, inner_iterations,
                             batch_size_alignement, folder=folder)#, x_axis=test_accuracies[algo_name])

    plot_values(train_epochs, train_accuracies, all_algos, 'Train accuracy', dataset_name, inner_iterations,
                batch_size_alignement, folder=folder)
    plot_values(train_epochs, train_losses, all_algos, 'log(Train loss)', dataset_name, inner_iterations,
                batch_size_alignement, folder=folder, log=True)
    plot_values(test_epochs, test_accuracies, all_algos, 'Test accuracy', dataset_name, inner_iterations,
                batch_size_alignement, folder=folder)
    plot_values(test_epochs, test_losses, all_algos, 'log(Test loss)', dataset_name, inner_iterations,
                batch_size_alignement, folder=folder, log=True)




