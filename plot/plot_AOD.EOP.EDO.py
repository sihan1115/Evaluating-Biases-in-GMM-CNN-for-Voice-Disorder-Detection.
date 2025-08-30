#!/usr/bin/env python
# coding: utf-8

"""
Fairness Metrics Visualization Script
This script creates bar charts with error bars to visualize fairness metrics (AOD, EOP, EOD)
across different models (GMM, CNN) and datasets (EENT, SVD) for gender or age dimensions.
The visualization includes error bars representing standard deviations and threshold lines.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import re
import os

# Configure font and plot style
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300

# ============ Configuration Parameters ======================
FILE_PATH = 'dataframe_for_figure.xlsx'  # Path to Excel data file
DIMENSION = 'gender'  # Dimension to analyze: 'age' or 'gender'
METRIC = 'EOD'  # Fairness metric to visualize: 'AOD', 'EOP', 'EOD'
THRESHOLD = 0.25  # Position of threshold line
CUSTOM_TITLE = 'EOD'  # Custom chart title: 'AOD', 'EOP', 'EOD'
SAVE_DIR = r"C:\Users\DELL\Desktop\加油小粽子\fairness图"  # Directory to save output figures
# ===========================================================

def parse_mean_std(value_str):
    """
    Parse mean±standard deviation formatted strings
    Example: "0.132±0.038" -> (0.132, 0.038)

    Parameters:
    value_str (str): String in format "mean±std" or numeric value

    Returns:
    tuple: (mean, standard_deviation) as floats
    """
    if pd.isna(value_str) or value_str == '':
        return 0.0, 0.0

    # If it's already a number rather than string, return directly
    if isinstance(value_str, (int, float)):
        return float(value_str), 0.0

    # Handle string format
    value_str = str(value_str)

    # Try to match "number±number" pattern
    pattern = r'([0-9.]+)±([0-9.]+)'
    match = re.search(pattern, value_str)

    if match:
        mean_val = float(match.group(1))
        std_val = float(match.group(2))
        return mean_val, std_val
    else:
        # If no ± format is matched, try to parse as single number
        try:
            mean_val = float(value_str)
            return mean_val, 0.0
        except:
            return 0.0, 0.0


def create_aod_plot(file_path=None):
    """
    Create fairness comparison chart with error bars for AOD/EOP/EOD metrics

    Parameters:
    file_path (str, optional): Path to data file. If None, uses FILE_PATH global variable
    """

    # Determine file path
    data_file = file_path if file_path else FILE_PATH

    # Load data
    try:
        df = pd.read_excel(data_file)
        print(f"Successfully loaded data file: {data_file}")
    except FileNotFoundError:
        print(f"Error: Cannot find file {data_file}")
        return
    except Exception as e:
        print(f"Error: Problem loading file - {e}")
        return

    # Filter data based on dimension and metric
    filtered_df = df[(df['dimension'] == DIMENSION) & (df['metric'] == METRIC)]

    # Create scenario label mapping
    scenario_mapping = {
        ('EENT', 'a'): 'a-EENT',
        ('EENT', 'i'): 'i-EENT',
        ('EENT', 'combined'): 'combined-EENT',
        ('SVD', 'a'): 'a-SVD',
        ('SVD', 'i'): 'i-SVD',
        ('SVD', 'combined'): 'combined-SVD'
    }

    # Prepare data for plotting
    scenarios = ['a-EENT', 'i-EENT', 'combined-EENT', 'a-SVD', 'i-SVD', 'combined-SVD']
    gmm_means = []
    gmm_stds = []
    cnn_means = []
    cnn_stds = []

    # Extract data for each scenario
    for scenario in scenarios:
        # Find corresponding testset and vowel
        testset_vowel = [(k, v) for (k, v), s in scenario_mapping.items() if s == scenario][0]
        testset, vowel = testset_vowel

        # Get GMM and CNN values
        gmm_row = filtered_df[(filtered_df['model'] == 'GMM') &
                              (filtered_df['testset'] == testset) &
                              (filtered_df['vowel'] == vowel)]
        cnn_row = filtered_df[(filtered_df['model'] == 'CNN') &
                              (filtered_df['testset'] == testset) &
                              (filtered_df['vowel'] == vowel)]

        # Parse GMM data
        if len(gmm_row) > 0:
            gmm_mean, gmm_std = parse_mean_std(gmm_row['value'].iloc[0])
        else:
            gmm_mean, gmm_std = 0.0, 0.0

        # Parse CNN data
        if len(cnn_row) > 0:
            cnn_mean, cnn_std = parse_mean_std(cnn_row['value'].iloc[0])
        else:
            cnn_mean, cnn_std = 0.0, 0.0

        gmm_means.append(gmm_mean)
        gmm_stds.append(gmm_std)
        cnn_means.append(cnn_mean)
        cnn_stds.append(cnn_std)

    print(f"GMM means: {gmm_means}")
    print(f"GMM standard deviations: {gmm_stds}")
    print(f"CNN means: {cnn_means}")
    print(f"CNN standard deviations: {cnn_stds}")

    # Create figure with larger size
    fig, ax = plt.subplots(figsize=(20, 12))

    # Set bar positions
    x = np.arange(len(scenarios)) * 1.1
    width = 0.4

    # Plot bars with error bars - customized colors
    bars1 = ax.bar(x - width / 2, gmm_means, width, yerr=gmm_stds,
                   label='GMM', color='#67A9CC', alpha=0.8,
                   edgecolor='#3A7CA5', linewidth=2,
                   capsize=8, error_kw={'linewidth': 3, 'capthick': 3})

    bars2 = ax.bar(x + width / 2, cnn_means, width, yerr=cnn_stds,
                   label='CNN', color='#DD9B26', alpha=0.8,
                   edgecolor='#B27F1A', linewidth=2,
                   capsize=8, error_kw={'linewidth': 3, 'capthick': 3})

    # Add threshold line
    ax.axhline(y=THRESHOLD, color='#e74c3c', linestyle='--', linewidth=4,
               alpha=1, label=f'threshold={THRESHOLD}')

    # Set chart properties - remove title, increase font sizes
    ax.set_xlabel('Model', fontsize=24, fontweight='bold')  # Increase font size
    ax.set_ylabel(f'{METRIC} (mean ± std)', fontsize=24, fontweight='bold')  # Increase font size
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=22, fontweight='bold')  # Increase font size

    # Dynamically set y-axis range
    all_values = gmm_means + cnn_means
    all_errors = gmm_stds + cnn_stds
    max_val = max([v + e for v, e in zip(all_values, all_errors)]) if all_values else 1

    # Ensure threshold line is visible
    upper_limit = max(max_val * 1.15, THRESHOLD * 1.2)
    ax.set_ylim(0, upper_limit)

    # Set y-axis ticks - increase font size
    ax.tick_params(axis='y', labelsize=20)

    # Add legend - increase font size
    ax.legend(fontsize=20, loc='upper left', bbox_to_anchor=(1, 1), framealpha=0.9)

    # Add value labels on bars (show mean±standard deviation)
    def add_value_labels(bars, means, stds):
        """
        Add value labels on top of bars showing mean±std values

        Parameters:
        bars: Bar objects to label
        means: List of mean values
        stds: List of standard deviation values
        """
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            error_top = height + std

            if std > 0:
                label_text = f'{mean:.3f}±{std:.3f}'
            else:
                label_text = f'{mean:.3f}±{std:.0f}'

            # Label position above error bar top
            ax.annotate(label_text,
                        xy=(bar.get_x() + bar.get_width() / 2, error_top),
                        xytext=(0, 8),  # Increase offset
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=16, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3",
                                  facecolor='white',
                                  edgecolor='gray',
                                  alpha=0.8))

    add_value_labels(bars1, gmm_means, gmm_stds)
    add_value_labels(bars2, cnn_means, cnn_stds)

    # Beautify chart
    ax.grid(True, alpha=0.3, axis='y', zorder=0)
    ax.set_axisbelow(True)

    # Adjust layout
    plt.tight_layout()

    # Create save directory if it doesn't exist
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Save figure to specified directory
    save_filename = os.path.join(SAVE_DIR, f'{DIMENSION}_{METRIC}_errorbar.png')
    plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    print(f"Figure saved as: {save_filename}")

    # Display chart
    # plt.show()


# Run plotting function
if __name__ == "__main__":
    print(f"Plotting: {DIMENSION} dimension {METRIC} metric comparison (mean±standard deviation)")
    print(f"Chart title: {CUSTOM_TITLE}")
    print(f"Threshold line: {THRESHOLD}")

    create_aod_plot()

