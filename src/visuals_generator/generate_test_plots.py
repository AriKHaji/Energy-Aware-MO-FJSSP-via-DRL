import matplotlib.pyplot as plt
from adjustText import adjust_text
import numpy as np

# Dictionary of label -> (makespan, total energy consumption)
results = {
    "SFJS01": {
        "RS1": (130, 694.73),
        "RS2": (130, 694.73),
        "RS3": (130, 694.73),
        "RS4": (130, 694.73),
        "RS5": (130, 694.73),
        "RS6": (130, 694.73),
        "RS7": (130, 694.73),
        "RS8": (110, 714.57),
        "MOR": (86, 767.71),
        "LOR": (86, 767.71),
        "SPT": (91, 804.01),
        "LEC": (191, 620.91),
        "rnd": (89, 737.43)
    },
    "SFJS02": {
        "RS1": (107, 771.87),
        "RS2": (107, 771.87),
        "RS3": (107, 771.87),
        "RS4": (107, 771.87),
        "RS5": (107, 771.87),
        "RS6": (107, 771.87),
        "RS7": (107, 771.87),
        "RS8": (128, 817.55),
        "MOR": (107, 771.87),
        "LOR": (107, 771.87),
        "SPT": (128, 817.55),
        "LEC": (157, 709.38),
        "rnd": (157, 754.22)
    },
    "SFJS03": {
        "RS1": (269, 1001.76),
        "RS2": (298, 1063.16),
        "RS3": (269, 1001.76),
        "RS4": (265, 1017.04),
        "RS5": (303, 989.54),
        "RS6": (303, 989.54),
        "RS7": (269, 1001.76),
        "RS8": (269, 1001.76),
        "MOR": (221, 983.1),
        "LOR": (304, 1018.6),
        "SPT": (298, 1063.16),
        "LEC": (417, 883.485),
        "rnd": (316, 990.06)
    },
    "SFJS04": {
        "RS1": (396, 821.3),
        "RS2": (409, 843.11),
        "RS3": (409, 843.11),
        "RS4": (409, 843.11),
        "RS5": (355, 779.6),
        "RS6": (409, 843.11),
        "RS7": (409, 843.11),
        "RS8": (396, 821.3),
        "MOR": (367, 764.45),
        "LOR": (409, 843.11),
        "SPT": (542, 827.14),
        "LEC": (689, 649.89),
        "rnd": (470, 739.4)
    },
    "SFJS05": {
        "RS1": (198, 1194.86),
        "RS2": (205, 1208.52),
        "RS3": (151, 1082.62),
        "RS4": (253, 963.53),
        "RS5": (241, 1235.23),
        "RS6": (134, 1127.81),
        "RS7": (155, 1153.46),
        "RS8": (134, 1107.78),
        "MOR": (143, 1107.57),
        "LOR": (152, 1102.31),
        "SPT": (176, 1129.03),
        "LEC": (253, 963.53),
        "rnd": (218, 1006.97)
    },
    "SFJS06": {
        "RS1": (380, 1472.03),
        "RS2": (520, 1454.03),
        "RS3": (447, 1331.56),
        "RS4": (517, 1572.56),
        "RS5": (367, 1440.96),
        "RS6": (370, 1411.63),
        "RS7": (370, 1536.33),
        "RS8": (380, 1508.33),
        "MOR": (360, 1470.23),
        "LOR": (427, 1541.46),
        "SPT": (397, 1598.06),
        "LEC": (470, 1214.83),
        "rnd": (730, 1406.93)
    },
    "SFJS07": {
        "RS1": (397, 1276.42),
        "RS2": (407, 1277.15),
        "RS3": (667, 1301.48),
        "RS4": (517, 1075.45),
        "RS5": (517, 1052.65),
        "RS6": (667, 1244.07),
        "RS7": (447, 1095.49),
        "RS8": (477, 1299.28),
        "MOR": (407, 1196.72),
        "LOR": (450, 1267.56),
        "SPT": (397, 1276.42),
        "LEC": (656, 978.42),
        "rnd": (605, 1281.76)
    },
    "SFJS08": {
        "RS1": (334, 1447.88),
        "RS2": (372, 1512.26),
        "RS3": (330, 1475.87),
        "RS4": (430, 1397.59),
        "RS5": (367, 1350.76),
        "RS6": (446, 1643.03),
        "RS7": (457, 1450.59),
        "RS8": (330, 1548.69),
        "MOR": (280, 1548.76),
        "LOR": (281, 1575.19),
        "SPT": (413, 1637.15),
        "LEC": (583, 1291.04),
        "rnd": (405, 1533.78)
    },
    "SFJS09": {
        "RS1": (297, 1659.26),
        "RS2": (350, 1578.93),
        "RS3": (240, 1477.33),
        "RS4": (317, 1614.46),
        "RS5": (300, 1423.23),
        "RS6": (290, 1570.15),
        "RS7": (220, 1582.85),
        "RS8": (280, 1568.63),
        "MOR": (220, 1574.63),
        "LOR": (327, 1684.06),
        "SPT": (307, 1725.96),
        "LEC": (470, 1335.55),
        "rnd": (350, 1594.75)
    },
    "SFJS10": {
        "RS1": (703, 1329.89),
        "RS2": (934, 1368.84),
        "RS3": (923, 1321.09),
        "RS4": (914, 1344.18),
        "RS5": (1148, 1366.42),
        "RS6": (934, 1312.28),
        "RS7": (934, 1418.56),
        "RS8": (772, 1438.39),
        "MOR": (617, 1309.03),
        "LOR": (826, 1501.65),
        "SPT": (716, 1332.45),
        "LEC": (1508, 1342.61),
        "rnd": (867, 1258.01)
    }
}

def plot_results(results):
    """
    Visualizes a collection of datasets by creating scatter plots of makespan vs total energy consumption.

    Each dataset in the `results` dictionary is expected to be a dictionary mapping label names to
    a tuple of (makespan, total energy consumption). A separate plot is created for each dataset.

    Parameters:
        results (dict): A dictionary where each key is a dataset name and the value is another dictionary
                        mapping labels to tuples (makespan, total energy consumption).
    """
    # List of distinct colors for plotting
    colors = [
        'red', 'blue', 'green', 'orange', 'purple',
        'brown', 'pink', 'gray', 'olive', 'cyan',
        'magenta', 'darkgreen', 'black'
    ]

    for dataset_name, dataset in results.items():
        plt.figure()
        texts = []  # Store text annotations for adjustment

        for i, (label, values) in enumerate(dataset.items()):
            if not values:
                continue

            makespan, energy = values
            color = colors[i % len(colors)]
            plt.scatter(makespan, energy, color=color)
            if label == "MOR":
                # Offset für MOR, um Überlappungen zu reduzieren
                txt = plt.text(makespan + 1, energy - 1, label, fontsize=10, ha='left', va='top')
            else:
                txt = plt.text(makespan, energy, label, fontsize=10, ha='left', va='bottom')
            texts.append(txt)

        # Adjust text labels to avoid overlapping
        adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray', shrinkA=10))
        plt.xlabel('Makespan')
        plt.ylabel('Total Energy Consumption')
        plt.title(dataset_name)
        plt.tight_layout()
        plt.show()

#plot_results({'SFJS01': results.get('SFJS01')})
# plot_results(results)

def plot_results_with_extended_line(results):
    """
    Plottet alle Punkte pro Datensatz (Scatter Plot) und berechnet zusätzlich
    eine mathematische Gerade, die durch die Punkte "MOR" und "LEC" verläuft und
    in beide Richtungen über die Punkte hinaus erweitert wird. Dabei wird kein Label
    für diese Gerade in der Legende angegeben.

    Parameter:
        results (dict): Dictionary, in dem jeder Schlüssel einen Datensatz und dessen Werte als
                        Dictionary mit Tupeln (makespan, energy) repräsentiert.
    """
    for dataset_name, dataset in results.items():
        plt.figure()
        texts = []  # Liste für Textannotationen

        # Scatter Plot aller Punkte
        for label, values in dataset.items():
            if not values:
                continue
            makespan, energy = values
            plt.scatter(makespan, energy)
            txt = plt.text(makespan, energy, f' {label}', fontsize=10, ha='left', va='bottom')
            texts.append(txt)

        # Berechne und zeichne die erweiterte Gerade, falls MOR und LEC vorhanden sind.
        if "MOR" in dataset and "LEC" in dataset:
            mor_makespan, mor_energy = dataset["MOR"]
            lec_makespan, lec_energy = dataset["LEC"]

            if abs(lec_makespan - mor_makespan) > 1e-7:
                m = (lec_energy - mor_energy) / (lec_makespan - mor_makespan)
                b = mor_energy - m * mor_makespan
                x_min = min(mor_makespan, lec_makespan)
                x_max = max(mor_makespan, lec_makespan)
                delta = (x_max - x_min) * 0.2  # 20% Extra in beide Richtungen
                x_vals = np.linspace(x_min - delta, x_max + delta, 100)
                y_vals = m * x_vals + b
            else:
                x_val = mor_makespan
                y_min = min(mor_energy, lec_energy)
                y_max = max(mor_energy, lec_energy)
                delta = (y_max - y_min) * 0.2
                y_vals = np.linspace(y_min - delta, y_max + delta, 100)
                x_vals = np.full_like(y_vals, x_val)

            # Zeichne die Gerade ohne Label, sodass kein zusätzlicher Eintrag in der Legende erscheint
            plt.plot(x_vals, y_vals, 'k-')

        adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray', shrinkA=10))
        plt.xlabel('Makespan')
        plt.ylabel('Total Energy Consumption')
        plt.title(dataset_name)
        plt.tight_layout()
        plt.show()
plot_results_with_extended_line(results)

def plot_entries_across_sfjs(results, keys_to_plot):
    """
    Plots entries for specific keys (e.g., ["RS1", "RS2", "RS3"]) across different sfjs datasets.

    For each sfjs dataset (each key in results), this function iterates through the provided
    keys_to_plot. If a key exists in the dataset, the corresponding point (with x = makespan,
    y = total energy consumption) is plotted using a marker specific to that sfjs. The point is
    also annotated with the key (e.g. "RS1") so you can see which algorithm it represents.

    Parameters:
        results (dict): A dictionary where each key is a dataset name (e.g., "SFJS01") and the
                        value is another dictionary mapping algorithm labels to tuples
                        (makespan, total energy consumption).
        keys_to_plot (list): A list of keys (algorithm labels) that should be plotted across all sfjs datasets.
    """
    # A list of markers to differentiate each sfjs dataset.
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'x', '+', 'h']

    plt.figure()
    texts = []  # List to store text annotations for later adjustment.
    added_sfjs = {}  # Dictionary to track if we have already added a legend label for a given sfjs.

    # Iterate over each sfjs dataset.
    for idx, (sfjs_name, dataset) in enumerate(results.items()):
        marker = markers[idx % len(markers)]
        added_sfjs[sfjs_name] = False  # Initialize flag for this sfjs dataset.
        for key in keys_to_plot:
            if key in dataset:
                makespan, energy = dataset[key]
                # On the first occurrence for a given sfjs dataset, add a legend label.
                if not added_sfjs[sfjs_name]:
                    plt.scatter(makespan, energy, marker=marker, label=sfjs_name)
                    added_sfjs[sfjs_name] = True
                else:
                    plt.scatter(makespan, energy, marker=marker)
                # Annotate the point with the key (e.g., "RS1").
                txt = plt.text(makespan, energy, key, fontsize=10, ha='left', va='bottom')
                texts.append(txt)

    # Adjust text annotations to reduce overlapping.
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray', shrinkA=10))

    plt.xlabel('Makespan')
    plt.ylabel('Total Energy Consumption')
    plt.title('Selected Entries Across sfjs')
    plt.legend()  # Legend maps markers to each sfjs dataset.
    plt.tight_layout()
    plt.show()

def plot_keys_across_sfjs_bars(results, keys_to_plot, title):
    """
    Creates two grouped bar charts (side by side):
      - Left chart compares 'makespan' across all requested keys, per sfjs dataset.
      - Right chart compares 'energy consumption' across the same keys, per sfjs dataset.

    Parameters:
        results (dict):
            A dictionary of the form:
                {
                    "SFJS01": {
                        "RS1": (makespan_value, energy_value),
                        "RS2": (makespan_value, energy_value),
                        ...
                    },
                    "SFJS02": {
                        ...
                    },
                    ...
                }
        keys_to_plot (list of str):
            The labels (e.g., "RS1", "RS2", "RS3", etc.) you want to compare across all sfjs datasets.
    """
    # Sort the sfjs dataset names to ensure consistent ordering on the x-axis
    dataset_names = sorted(results.keys())

    # Prepare data structures to hold the makespan and energy values for each key
    makespan_data = {key: [] for key in keys_to_plot}
    energy_data   = {key: [] for key in keys_to_plot}

    # Gather data for each sfjs dataset
    for sfjs_name in dataset_names:
        dataset = results[sfjs_name]
        for key in keys_to_plot:
            if key in dataset:
                ms, en = dataset[key]
                makespan_data[key].append(ms)
                energy_data[key].append(en)
            else:
                # If the key does not exist for this sfjs, use NaN (ignored by bar chart)
                makespan_data[key].append(np.nan)
                energy_data[key].append(np.nan)

    # Number of dataset categories on x-axis
    x = np.arange(len(dataset_names))

    # Width of each bar (we'll group them side by side for each sfjs dataset)
    bar_width = 0.8 / len(keys_to_plot)

    # Set up the figure with two subplots: one for makespan, one for energy
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(title, fontsize=16)

    # ---- Makespan subplot (ax1) ----
    for i, key in enumerate(keys_to_plot):
        shift = i * bar_width  # Shift for each key so bars don’t overlap
        ax1.bar(x + shift,
                makespan_data[key],
                bar_width,
                label=key)

    ax1.set_xlabel("SFJS Dataset")
    ax1.set_ylabel("Makespan")
    # Position x-ticks in the middle of the group
    ax1.set_xticks(x + (len(keys_to_plot)-1)*bar_width / 2)
    ax1.set_xticklabels(dataset_names, rotation=45, ha="right")
    ax1.legend(title="Reward Strategies")

    # ---- Energy consumption subplot (ax2) ----
    for i, key in enumerate(keys_to_plot):
        shift = i * bar_width
        ax2.bar(x + shift,
                energy_data[key],
                bar_width,
                label=key)

    ax2.set_xlabel("SFJS Dataset")
    ax2.set_ylabel("Total Energy Consumption")
    ax2.set_xticks(x + (len(keys_to_plot)-1)*bar_width / 2)
    ax2.set_xticklabels(dataset_names, rotation=45, ha="right")

    # (Optional) If you want a legend on both subplots, remove the comment below:
    # ax2.legend(title="Key (Algorithm)")

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the suptitle
    plt.show()

# plot_keys_across_sfjs_bars(results, ["RS1", "RS2", "RS3"], "Comparison of Weighted Sum Approaches across SFJS Datasets")
# plot_keys_across_sfjs_bars(results, ["RS3", "RS4", "RS5", "RS6"], "Comparison of Different Makespan and Energy Consumption Reward Metrics")
# plot_keys_across_sfjs_bars(results, ["RS3", "RS7", "RS8"], "Comparison based on Feedback Density across SFJS Datasets")

