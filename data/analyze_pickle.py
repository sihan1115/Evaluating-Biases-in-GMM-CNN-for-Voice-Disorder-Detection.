# Script to analyze pickle files and extract detailed information such as patient IDs, sample counts, and label distributions.
# Generates summary statistics and visualizations, and saves results to an Excel file.
import os
import joblib
import collections
import matplotlib.pyplot as plt
import pandas as pd


def analyze_single_pkl(pkl_path):
    """
    Analyze a single pickle file to extract statistics.

    Args:
        pkl_path (str): Path to the pickle file

    Returns:
        dict: Dictionary containing filename, total samples, unique speakers,
              label counts, and ID counts
    """
    with open(pkl_path, 'rb') as f:
        data = joblib.load(f)

    features = data['features']
    labels = data['labels']
    ids = data['ids']

    total_samples = len(features)
    unique_speakers = len(set(ids))
    label_counter = collections.Counter(labels)
    id_counter = collections.Counter(ids)

    return {
        'filename': os.path.basename(pkl_path),
        'total_samples': total_samples,
        'unique_speakers': unique_speakers,
        'label_counts': label_counter,
        'id_counts': id_counter
    }


def analyze_all_pickles(pkl_folder, save_excel_path="summary_stats.xlsx", show_plot=True):
    """
    Analyze all pickle files in a folder and generate summary statistics.

    Args:
        pkl_folder (str): Path to folder containing pickle files
        save_excel_path (str): Path where to save the Excel output file
        show_plot (bool): Whether to display the label distribution plot

    Returns:
        pandas.DataFrame: Summary statistics dataframe
    """
    # Find all pickle files in the specified folder
    pkl_files = [f for f in os.listdir(pkl_folder) if f.endswith('.pkl')]
    if not pkl_files:
        print("No .pkl files found.")
        return

    results = []
    label_set = set()
    all_id_dfs = {}  # Store all ID distribution DataFrames

    # Process each pickle file
    for f in pkl_files:
        full_path = os.path.join(pkl_folder, f)
        print(f"Analyzing: {f}")
        result = analyze_single_pkl(full_path)
        label_set.update(result['label_counts'].keys())
        results.append(result)

        # Create ID distribution table for this file
        id_df = pd.DataFrame(result['id_counts'].items(), columns=['ID', 'Sample Count'])
        id_df = id_df.sort_values(by='Sample Count', ascending=False).reset_index(drop=True)
        all_id_dfs[f] = id_df

    # Build summary table with all labels
    label_list = sorted(list(label_set))
    summary = []
    for r in results:
        row = {
            'filename': r['filename'],
            'total_samples': r['total_samples'],
            'unique_speakers': r['unique_speakers']
        }
        for l in label_list:
            row[f'label_{l}'] = r['label_counts'].get(l, 0)
        summary.append(row)

    df_summary = pd.DataFrame(summary)
    print("\nSummary Table:")
    print(df_summary)

    # Visualize label distribution across all files
    if show_plot:
        df_plot = df_summary.set_index('filename')
        ax = df_plot[['label_' + str(l) for l in label_list]].plot(
            kind='bar', stacked=True, figsize=(12, 5), colormap='viridis')
        plt.title("Label Distribution in All Pickle Files")
        plt.xlabel("Pickle File")
        plt.ylabel("Sample Count")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    # Save all results to Excel file with separate sheets
    with pd.ExcelWriter(save_excel_path) as writer:
        df_summary.to_excel(writer, sheet_name="summary_stats", index=False)
        for fname, id_df in all_id_dfs.items():
            sheet_name = os.path.splitext(fname)[0]
            # Excel sheet names are limited to 31 characters
            id_df.to_excel(writer, sheet_name=sheet_name[:31], index=False)

    print(f"\nExcel saved to: {save_excel_path}")
    return df_summary


# Main execution - analyze all pickle files in the specified folder
pkl_folder = "D:/data/audio/final_pickle_files"
analyze_all_pickles(pkl_folder, save_excel_path="D:/data/audio/final_pickle_files/summary_stats_with_ids.xlsx")
