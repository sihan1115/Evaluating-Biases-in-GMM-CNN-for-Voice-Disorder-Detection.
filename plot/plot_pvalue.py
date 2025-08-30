#!/usr/bin/env python
# coding: utf-8

"""
Statistical Significance Visualization Script
This script creates bar charts to visualize the proportion of p-values < 0.05 from chi-square tests
for accuracy or F1-score metrics across different models (GMM, CNN) and datasets (EENT, SVD) 
for age or gender dimensions. This helps assess statistical significance of performance differences.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# Configure font and plot style
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300

# ============ Configuration Parameters ======================
FILE_PATH = 'dataframe_for_figure.xlsx'  # Path to Excel data file
DIMENSION = 'age'  # Dimension to analyze: 'age' or 'gender'
METRIC = 'accuracy_p'  # Metric to visualize: 'accuracy_p' or 'F1 score_p'
CUSTOM_TITLE = 'Accuracy'  # Custom chart title: 'Accuracy' or 'F1 score'
SAVE_DIR = r"C:\Users\DELL\Desktop\加油小粽子\fairness图"  # Directory to save output figures
# ===========================================================

def create_fairness_plot(file_path=None):
    """
    Create statistical significance comparison chart
    Visualizes the proportion of p-values < 0.05 from chi-square tests for accuracy or F1-score

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
    gmm_values = []
    cnn_values = []

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

        # Convert proportion to percentage
        gmm_val = gmm_row['value'].iloc[0] * 100 if len(gmm_row) > 0 else 0
        cnn_val = cnn_row['value'].iloc[0] * 100 if len(cnn_row) > 0 else 0

        gmm_values.append(gmm_val)
        cnn_values.append(cnn_val)

    # Create figure with larger size to accommodate larger text
    fig, ax = plt.subplots(figsize=(20, 12))

    # Set bar positions with increased spacing between scenario groups
    x = np.arange(len(scenarios)) * 1.1
    width = 0.4

    # Plot bars - customized colors
    bars1 = ax.bar(x - width / 2, gmm_values, width, label='GMM',
                   color='#67A9CC', alpha=0.8, edgecolor='#3A7CA5', linewidth=2)
    bars2 = ax.bar(x + width / 2, cnn_values, width, label='CNN',
                   color='#DD9B26', alpha=0.8, edgecolor='#B27F1A', linewidth=2)

    # Add 50% threshold line (reference line for interpretation)
    ax.axhline(y=50, color='#e74c3c', linestyle='--', linewidth=4, alpha=1, label='threshold=50%')

    # Set chart properties
    ax.set_xlabel('Model', fontsize=24, fontweight='bold')
    ax.set_ylabel('Accuracy chi-square test (proportion of p-values < 0.05)', fontsize=24,
                  fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=22, fontweight='bold')
    ax.set_ylim(0, 100)

    # Set y-axis ticks
    ax.set_yticks(np.arange(0, 101, 10))
    ax.tick_params(axis='y', labelsize=20)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0f}%'))

    # Add legend
    ax.legend(fontsize=20, loc='upper left', bbox_to_anchor=(1, 1), framealpha=0.9)

    # Add value labels on bars
    def add_value_labels(bars, values):
        """
        Add value labels on top of bars

        Parameters:
        bars: Bar objects to label
        values: List of values to display
        """
        for bar, value in zip(bars, values):
            height = bar.get_height()

            # Modify annotation display: for 0% don't show '%', and bold display '0'
            ax.annotate(
                f'{value:.0f}%' if value != 0 else '0',  # If value is 0, don't show percent sign
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),  # Increase vertical offset
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=18, fontweight='normal' if value != 0 else 'bold')

    add_value_labels(bars1, gmm_values)
    add_value_labels(bars2, cnn_values)

    # Beautify chart
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)

    # Adjust layout
    plt.tight_layout()

    # Create save directory if it doesn't exist
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Save figure to specified directory
    save_filename = os.path.join(SAVE_DIR, f'{DIMENSION}_{METRIC.replace(" ", "_")}.png')
    plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    print(f"Figure saved as: {save_filename}")


# Run plotting function
if __name__ == "__main__":
    print(f"Plotting: {DIMENSION} dimension {METRIC} metric comparison chart")
    print(f"Chart title: {CUSTOM_TITLE}")

    create_fairness_plot()
