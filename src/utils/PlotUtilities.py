import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

# Configure matplotlib to use LaTeX for all text rendering
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

from src.data.DatasetConstants import BATCH_SIZE, STEP_SIZE, MOMENTUM, NB_CLIENTS, SPLIT, SCHEDULER_PARAMS
from src.utils.Utilities import get_project_root, create_folder_if_not_existing

# Default plotting styles
COLORS_ALGO = {"All-for-one-bin": 'tab:blue', "All-for-one-cont": 'tab:red', "All-for-one-opt": 'tab:green',
          "Local": 'tab:orange', "FedAvg": 'tab:brown', "Ditto": 'tab:purple', "Cobo": 'tab:cyan',
          "Wga-bc": 'tab:grey', "Apfl": 'tab:pink'}
COLORS = ['tab:blue', 'tab:red', 'tab:orange', 'tab:brown', 'tab:green', 'tab:purple', 'tab:cyan', 'tab:pink',
          'tab:grey']

MARKERS = ['o', 's', 'D', '^', 'v', '<', 'P', 'X']
FONTSIZE = 20

def plot_values(epochs, values, legends, metric_name: str, dataset_name: str, inner_iterations: int, batch_size_alignement: int,
                folder = "", log=False):
    """
    Plot the mean and standard deviation of a metric over epochs for multiple algorithms.

    Args:
        epochs (dict): Dictionary of lists of epoch numbers for each algorithm (usually same x-axis, e.g., {"Local": [0, ..., T]}).
        values (dict): Dictionary where each key is an algorithm name and value is a list of metric trajectories (2D list).
        legends (list): List of algorithm names to plot (must match keys in `values`).
        metric_name (str): Name of the metric to display on the y-axis (e.g., "Test accuracy").
        dataset_name (str): Dataset used (used to determine save path and legend position).
        log (bool): If True, apply log10 transform to the values before averaging and plotting.
    """
    fig, ax = plt.subplots(figsize=(9, 7))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)
    plt.ylabel(metric_name, fontsize=FONTSIZE)
    plt.xlabel("Number of epochs", fontsize=FONTSIZE)
    if dataset_name in ["heart_disease", "ixi"]:
        if "accuracy" in metric_name:
            axins = zoomed_inset_axes(ax, 15 if dataset_name == "heart_disease" else 20, loc=4)
        else:
            axins = zoomed_inset_axes(ax, 50 if dataset_name == "heart_disease" else 10, loc=1)

    min_avg = np.inf
    max_avg = -np.inf

    i = 0
    # Plot each algorithm's mean and std across runs
    for algo_name in legends:
        value_to_plot = []
        for s in values[algo_name].keys():
            for l in values[algo_name][s]:
                value_to_plot.append(l)
        if log:
            avg_values = np.mean([np.log10(v) for v in value_to_plot], axis=0)
            avg_values_var = np.std([np.log10(v) for v in value_to_plot], axis=0)
        else:
            avg_values = np.mean(value_to_plot, axis=0)
            avg_values_var = np.std(value_to_plot, axis=0)
        if avg_values[-1] < min_avg:
            min_avg = avg_values[-1]
        if avg_values[-1] > max_avg:
            max_avg = avg_values[-1]


        epochs_axis = np.linspace(0, len(avg_values)-1, len(avg_values))
        ax.plot(epochs_axis, avg_values, linestyle='-', color=COLORS_ALGO[algo_name], label=algo_name, linewidth=5)
        ax.fill_between(epochs_axis, avg_values - avg_values_var, avg_values + avg_values_var, alpha=0.2,
                         color=COLORS_ALGO[algo_name])

        if dataset_name in ["heart_disease", "ixi"]:
            axins.plot(epochs_axis, avg_values, linestyle='-', color=COLORS_ALGO[algo_name], label=algo_name, linewidth=5)
            if "accuracy" in metric_name:
                if dataset_name == "ixi":
                    x1, x2, y1, y2 = epochs_axis[-1]-0.25, epochs_axis[-1], max_avg-0.003, max_avg+0.002
                elif dataset_name == "heart_disease":
                    x1, x2, y1, y2 = epochs_axis[-1]-0.25, epochs_axis[-1], max_avg-0.01, max_avg+0.005
            else:
                if dataset_name == "ixi":
                    x1, x2, y1, y2 = epochs_axis[-1] - 0.25, epochs_axis[-1], min_avg - 0.005, min_avg + 0.03
                elif dataset_name == "heart_disease":
                    x1, x2, y1, y2 = epochs_axis[-1] - 0.25, epochs_axis[-1], min_avg - 0.0005, min_avg + 0.001
            axins.set_xlim(x1, x2)
            axins.set_ylim(y1, y2)
            axins.set_xticks([])
            axins.set_yticks([])
            if "accuracy" in metric_name:
                mark_inset(ax, axins, loc1=1, loc2=2, ec="black")
            else:
                mark_inset(ax, axins, loc1=3, loc2=4, ec="black")

        i += 1

    # Heuristic for legend placement depending on dataset and metric
    if (metric_name == "log(Test loss)" and dataset_name == "mnist"):
        loc = "lower left"
    elif (metric_name == "Test accuracy" and dataset_name == "mnist"):
        loc = "lower right"
    else:
        loc = "lower left"

    if dataset_name in ["mnist", "synth"]:
        plt.legend(fontsize=FONTSIZE, loc="best", ncol=2)

    # Save figure to disk
    root = get_project_root()
    folder = f'{root}/pictures/{folder}/{dataset_name}'
    create_folder_if_not_existing(folder)
    if dataset_name in SPLIT.keys():
        ID = (f"{metric_name}_N{NB_CLIENTS[dataset_name]}_b{BATCH_SIZE[dataset_name]}_LR{STEP_SIZE[dataset_name]}_"
              f"s{SCHEDULER_PARAMS[dataset_name][0]}_m{MOMENTUM[dataset_name]}_inner{inner_iterations}_"
              f"bAl{batch_size_alignement}_{SPLIT[dataset_name]}")
    else:
        ID = (f"{metric_name}_N{NB_CLIENTS[dataset_name]}_b{BATCH_SIZE[dataset_name]}_LR{STEP_SIZE[dataset_name]}_"
              f"s{SCHEDULER_PARAMS[dataset_name][0]}_m{MOMENTUM[dataset_name]}_inner{inner_iterations}_"
              f"bAl{batch_size_alignement}_")
    plt.savefig(f"{folder}/{ID}.pdf", bbox_inches='tight', dpi=600)

    # Print final metric value (e.g., accuracy or log-loss) in LaTeX tabular format for paper inclusion
    print("\\begin{tabular}{|c|c|}")
    print("\\hline")
    print(f"Algorithm & {metric_name} \\\\")
    print("\\hline")
    for algo_name in legends:
        value_to_plot = []
        for s in values[algo_name].keys():
            for l in values[algo_name][s]:
                value_to_plot.append(l)
        if log:
            final_value = np.mean([np.log10(v) for v in value_to_plot], axis=0)[-1]
        else:
            final_value = np.mean(value_to_plot, axis=0)[-1]
        print(f"{algo_name} & {final_value:.4f} \\\\")
    print("\\hline")
    print("\\end{tabular}")


def plot_weights(weights, dataset_name, algo_name, inner_iterations: int, batch_size_alignement: int,
                 folder = "", name="weights", x_axis=None):
    """
    Plot the evolution of per-client weights over training rounds.

    Args:
        weights (list): A list of lists: weights[i][t][j] is the weight client i assigns to client j at round t.
        dataset_name (str): Name of the dataset (used in path for saving).
        algo_name (str): Name of the algorithm (used in filename).
        name (str): Optional filename suffix.
        x_axis (list or None): Optional x-axis values per client (same shape as weights[i][t]).
    """
    nb_clients = len(weights)
    fig, axes = plt.subplots(min(nb_clients, len(MARKERS)), 1, figsize=(6, min(nb_clients, len(MARKERS))))

    # Plot each client’s perspective on others’ weights
    for client_idx in range(min(nb_clients, len(MARKERS))):
        if nb_clients > 1:
            ax = axes[client_idx]
        else:
            ax = axes
        weight = weights[client_idx]
        iterations = range(len(weight))  # x-axis if no custom x_axis

        for c_idx in range(min(nb_clients, len(MARKERS))):
            # Track how client i allocates weight to client j over time
            weight_to_plot = [weight[t][c_idx] for t in iterations]
            if x_axis:
                # Optional reordering for aligned x_axis
                sorted_indices = np.argsort(x_axis[c_idx][:-1])
                sorted_x_axis = np.array(x_axis[c_idx][:-1])[sorted_indices]
                sorted_weight_to_plot = np.array(weight_to_plot)[sorted_indices]
                ax.plot(np.log10(sorted_x_axis), sorted_weight_to_plot, label=f"Client i ← {c_idx}",
                        color=COLORS[c_idx], alpha=0.7, linestyle='-', marker=MARKERS[c_idx])
            else:
                ax.plot(iterations, weight_to_plot, label=f"Client i ← {c_idx}",
                        color=COLORS[c_idx], alpha=0.7, linestyle='-', marker=MARKERS[c_idx])

        ax.grid(True, linestyle='--', alpha=0.6)
        if client_idx == 0:
            ax.legend(ncol=2, fontsize=FONTSIZE, loc="best")

    plt.subplots_adjust(wspace=0, hspace=0)  # Remove vertical space between plots

    # Save the resulting plot to disk
    root = get_project_root()
    folder = f'{root}/pictures/{folder}/{dataset_name}'
    create_folder_if_not_existing(folder)
    if dataset_name in SPLIT.keys():
        ID = (f"N{NB_CLIENTS[dataset_name]}_b{BATCH_SIZE[dataset_name]}_LR{STEP_SIZE[dataset_name]}_"
              f"s{SCHEDULER_PARAMS[dataset_name][0]}_m{MOMENTUM[dataset_name]}_inner{inner_iterations}_"
              f"bAl{batch_size_alignement}_{SPLIT[dataset_name]}")
    else:
        ID = (f"N{NB_CLIENTS[dataset_name]}_b{BATCH_SIZE[dataset_name]}_LR{STEP_SIZE[dataset_name]}_"
              f"s{SCHEDULER_PARAMS[dataset_name][0]}_m{MOMENTUM[dataset_name]}_inner{inner_iterations}_"
              f"bAl{batch_size_alignement}")
    plt.savefig(f"{folder}/{algo_name}_{name}_{ID}.pdf", bbox_inches='tight', dpi=600)
