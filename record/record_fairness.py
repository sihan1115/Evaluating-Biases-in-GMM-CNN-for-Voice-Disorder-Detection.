# This script extracts fairness metrics from TXT files and saves them to an Excel file.
# It processes 8 fairness metrics across multiple TXT files and organizes them in groups.

import os
import re
import pandas as pd
from collections import defaultdict

# Define the 8 metrics we want to extract
METRICS = [
    "全局准确率p-value",
    "FPR p-value",
    "FNR p-value",
    "F1 score p-value",
    "AOD",
    "DI",
    "EOP",
    "EOD"
]


def extract_metric_values(file_path):
    """
    Extract all metric values from TXT file, preserving nan values
    Args:
        file_path (str): Path to the TXT file
    Returns:
        list: List of tuples containing (metric_name, value_str)
    """
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        content = file.read()

    # Use regular expressions to match all metric values, including nan
    pattern = r'(' + '|'.join(re.escape(metric) for metric in METRICS) + r'):\s*([0-9.]+|nan)'
    matches = re.findall(pattern, content, re.IGNORECASE)

    return matches


def parse_value(value_str):
    """
    Parse value string, handling nan cases
    Args:
        value_str (str): String representation of value
    Returns:
        float or str: Parsed value as float or "nan" string
    """
    if value_str.lower() == 'nan':
        return "nan"  # Return string "nan" directly
    try:
        return float(value_str)  # Convert numeric values to float
    except ValueError:
        return "nan"  # Return string "nan" on conversion failure


def process_file(file_path, num_groups=12):
    """
    Process a single file and extract all metric groups
    Args:
        file_path (str): Path to the TXT file
        num_groups (int): Number of groups to extract
    Returns:
        list: List of groups, each containing 8 metric values
    """
    matches = extract_metric_values(file_path)
    all_groups = []

    # Collect metric values by group (8 metrics per group)
    for i in range(0, len(matches), len(METRICS)):
        group = matches[i:i + len(METRICS)]
        group_values = []

        # Process each metric value
        for metric_name, value_str in group:
            value = parse_value(value_str)
            group_values.append(value)

        all_groups.append(group_values)

    # Ensure we have enough groups
    while len(all_groups) < num_groups:
        all_groups.append(["nan"] * len(METRICS))  # Fill missing groups with "nan" strings

    # Only take first num_groups groups
    return all_groups[:num_groups]


def main():
    """Main function to process all TXT files and extract fairness metrics"""
    # Configuration paths
    folder_path = input("Please enter the folder path containing TXT files: ").strip()
    output_excel = 'performance_metrics.xlsx'  # Output Excel file name

    # Validate folder path
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist")
        return

    # Get all TXT files
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

    if not txt_files:
        print(f"No TXT files found in '{folder_path}'")
        return

    print(f"Found {len(txt_files)} TXT files, starting processing...")

    # Create data storage structure
    data = defaultdict(list)

    # Add Performance Metrics column - repeat 12 times with 8 metrics each
    for _ in range(12):
        for metric in METRICS:
            data['Performance Metrics'].append(metric)

    # Process each file
    for filename in txt_files:
        file_path = os.path.join(folder_path, filename)
        col_name = os.path.splitext(filename)[0]

        print(f"Processing file: {filename}")
        try:
            metrics_groups = process_file(file_path)

            # Add metric values to data column (in order)
            for group in metrics_groups:
                for value in group:
                    data[col_name].append(value)

            print(f"  Successfully extracted {len(metrics_groups)} groups of metrics")
        except Exception as e:
            print(f"  Error processing file {filename}: {str(e)}")
            # Fill with "nan" on error
            for _ in range(12 * len(METRICS)):
                data[col_name].append("nan")

    # Create DataFrame
    df = pd.DataFrame(data)

    # Save to Excel
    df.to_excel(output_excel, index=False)

    print(f"\nProcessing completed! Results saved to: {output_excel}")
    print(f"Output format: {len(txt_files)} columns (files) + 1 column (metric names)")
    print(f"Total rows: {len(df)} rows (12 groups × 8 metrics = 96 rows/file)")

    # Validate data extraction
    if len(txt_files) > 0:
        sample_col = os.path.splitext(txt_files[0])[0]
        print(f"\nSample data verification ({sample_col} column):")
        for i in range(min(16, len(df))):  # Show first 16 rows
            metric = df['Performance Metrics'].iloc[i]
            value = df[sample_col].iloc[i]
            print(f"  Row {i + 1}: {metric} = {value}")


if __name__ == "__main__":
    main()
