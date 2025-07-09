import glob
import os

import numpy as np

from src.data.DatasetConstants import SPLIT, BATCH_SIZE, NB_CLIENTS, STEP_SIZE, MOMENTUM, SCHEDULER_PARAMS, \
    RUNNING_CLIENTS, BS_ALIGNEMENT, INNER_ITERATION
from src.data.Network import get_network
from src.plot.PlotMetrics import extract_number, my_dict
from src.utils.LoggingWriter import LoggingWriter
from src.utils.PlotUtilities import plot_values, plot_weights
from src.utils.Utilities import get_project_root, create_folder_if_not_existing

"""
Automatically generates Table 2 included in the paper. 
Report the final train and test loss/accuracy (average over clients).
"""

if __name__ == '__main__':
    all_algos = ["All-for-one-bin", "All-for-one-cont", "Local", "FedAvg", "Ditto", "Cobo", "Wga-bc", "Apfl"]
    all_dataset = ["mnist", "cifar10", "heart_disease", "ixi"]
    all_seeds = [127, 496, 1729]  # Mersenne number, Perfect number, Ramanujan number
    folder = "final"

    str_acc = ""
    str_loss = ""

    for i in range(len(all_algos)):
        algo_name = all_algos[i]

        if i == 0:
            str_acc += "Accuracy &"
            str_loss += "Loss &"
        elif i == 1:
            str_acc += "(in $\%$) &"
            str_loss += "(in log) &"
        else:
            str_acc += " & "
            str_loss += " & "

        str_acc += f" {algo_name} "
        str_loss += f" {algo_name} "

        assert algo_name in ["All-for-one-bin", "All-for-one-cont", "All-for-one-opt", "Local", "FedAvg",
                             "Ditto", "Cobo", "Wga-bc", "Apfl"], "Algorithm not recognized."
        print(f"--- ================== ALGO: {algo_name} ================== ---")

        train_epochs, train_losses, train_accuracies = {algo: [] for algo in all_dataset}, {algo: [] for algo in all_dataset}, {algo: [] for algo in all_dataset}
        test_epochs, test_losses, test_accuracies = {algo: [] for algo in all_dataset}, {algo: [] for algo in all_dataset}, {algo: [] for algo in all_dataset}

        for dataset_name in all_dataset:

            assert dataset_name in ["exam_llm", "mnist", "mnist_iid", "cifar10", "cifar10_iid", "heart_disease",
                                    "tcga_brca", "ixi", "liquid_asset",
                                    "synth", "synth_complex"], "Dataset not recognized."
            print(f"### ================== DATASET: {dataset_name} ================== ###")

            for seed in all_seeds:
                root = get_project_root()
                pickle_folder = f'{root}/pickle/{folder}/{dataset_name}/{algo_name}/{seed}'


                # Use glob to find all files matching the pattern
                file_pattern = os.path.join(pickle_folder, 'logging_writer*.pkl')
                matching_files = glob.glob(file_pattern)

                # Use glob to find all files matching the pattern
                if dataset_name in ["mnist", "cifar10"]:
                    split_type = SPLIT[dataset_name]
                    regex = f'logging_writer_*_N{NB_CLIENTS[dataset_name]}_' \
                            f'b{BATCH_SIZE[dataset_name]}_LR{STEP_SIZE[dataset_name]}_' \
                            f's{SCHEDULER_PARAMS[dataset_name][0]}_m{MOMENTUM[dataset_name]}_' \
                            f'inner{INNER_ITERATION[dataset_name]}_bAl{BS_ALIGNEMENT[dataset_name]}_' \
                            f'{split_type}.pkl'
                    file_pattern = os.path.join(pickle_folder, regex)
                else:
                    regex = f'logging_writer_*_N{NB_CLIENTS[dataset_name]}_' \
                            f'b{BATCH_SIZE[dataset_name]}_LR{STEP_SIZE[dataset_name]}_' \
                            f's{SCHEDULER_PARAMS[dataset_name][0]}_m{MOMENTUM[dataset_name]}_' \
                            f'inner{INNER_ITERATION[dataset_name]}_bAl{BS_ALIGNEMENT[dataset_name]}.pkl'
                    file_pattern = os.path.join(pickle_folder, regex)
                matching_files = glob.glob(file_pattern)

                # Extract the file names from the full paths
                file_names = sorted([os.path.basename(file) for file in matching_files
                                     if os.path.basename(file) != "logging_writer_central.pkl"], key=extract_number)
                if len(file_names) == 0:
                    raise ValueError(f"There is no corresponding files in {pickle_folder} for:\n {regex}")
                for name in file_names[:RUNNING_CLIENTS]:

                    writer = LoggingWriter.load(pickle_folder, name)

                    train_epochs[dataset_name].append(writer.retrieve_information("train_accuracy")[0])
                    train_accuracies[dataset_name].append(writer.retrieve_information("train_accuracy")[1])
                    train_losses[dataset_name].append(writer.retrieve_information("train_loss")[1])

                    test_epochs[dataset_name].append(writer.retrieve_information("test_accuracy")[0])
                    test_accuracies[dataset_name].append(writer.retrieve_information("test_accuracy")[1])
                    test_losses[dataset_name].append(writer.retrieve_information("test_loss")[1])


            train_acc = np.mean(train_accuracies[dataset_name], axis=0)
            test_acc = np.mean(test_accuracies[dataset_name], axis=0)
            str_acc += f"& {train_acc[-1]*100:.1f} & {test_acc[-1]*100:.1f} "

            train_lo = np.mean([np.log10(v) for v in train_losses[dataset_name]], axis=0)
            test_lo = np.mean([np.log10(v) for v in test_losses[dataset_name]], axis=0)
            str_loss += f"& {train_lo[-1]:.2f} & {test_lo[-1]:.2f} "

        str_acc += "\\\\ \n"
        str_loss += "\\\\ \n"
    tabular = str_acc + str_loss
    print(tabular)





