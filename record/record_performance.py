# This script extracts performance metrics from TXT files and saves them to an Excel file.
# It processes test performance metrics including Accuracy, Precision, Recall, Specificity, Sensitivity, F1-Score, and AUC from multiple TXT files.

import os
import re
import pandas as pd
from collections import defaultdict


def extract_all_metrics(file_path):
    """
    Extract all Test Performance Metrics blocks with 7 metrics from TXT file
    Args:
        file_path (str): Path to the TXT file
    Returns:
        list: List of tuples containing the 7 metrics for each Test Performance Metrics block
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Split text and search for all lines containing "Test Performance Metrics"
    lines = content.split('\n')
    metrics_list = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Check if line contains Test Performance Metrics
        if 'Test Performance Metrics:' in line:
            # Search forward for the next few lines to find 7 metrics
            accuracy = None
            precision = None
            recall = None
            specificity = None
            sensitivity = None
            f1_score = None
            auc = None

            # Look at next 20 lines maximum to find metrics
            for j in range(i + 1, min(i + 21, len(lines))):
                current_line = lines[j].strip()

                # Stop searching if we encounter next Performance Metrics block
                if 'Performance Metrics:' in current_line:
                    break

                # Search for each metric (case-insensitive matching)
                accuracy_match = re.search(r'Accuracy:\s*([0-9.]+)', current_line, re.IGNORECASE)
                if accuracy_match:
                    accuracy = float(accuracy_match.group(1))

                precision_match = re.search(r'Precision:\s*([0-9.]+)', current_line, re.IGNORECASE)
                if precision_match:
                    precision = float(precision_match.group(1))

                recall_match = re.search(r'Recall:\s*([0-9.]+)', current_line, re.IGNORECASE)
                if recall_match:
                    recall = float(recall_match.group(1))

                specificity_match = re.search(r'Specificity:\s*([0-9.]+)', current_line, re.IGNORECASE)
                if specificity_match:
                    specificity = float(specificity_match.group(1))

                sensitivity_match = re.search(r'Sensitivity:\s*([0-9.]+)', current_line, re.IGNORECASE)
                if sensitivity_match:
                    sensitivity = float(sensitivity_match.group(1))

                f1_match = re.search(r'F1-Score:\s*([0-9.]+)', current_line, re.IGNORECASE)
                if f1_match:
                    f1_score = float(f1_match.group(1))

                auc_match = re.search(r'AUC:\s*([0-9.]+)', current_line, re.IGNORECASE)
                if auc_match:
                    auc = float(auc_match.group(1))

            metrics_list.append((accuracy, precision, recall, specificity, sensitivity, f1_score, auc))
            print(f"  Found Test Performance Metrics group {len(metrics_list)}:")
            print(f"    Accuracy={accuracy}, Precision={precision}, Recall={recall}")
            print(f"    Specificity={specificity}, Sensitivity={sensitivity}, F1-Score={f1_score}, AUC={auc}")

        i += 1

    return metrics_list


def main():
    """Main function to process all TXT files and extract performance metrics"""
    # Configuration paths - directly specify folder path in code
    folder_path = r"C:\Users\DELL\Desktop\加油小粽子\final_output\CNN评估" # Please modify to your actual folder path
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

    # Create data storage structure
    data = defaultdict(list)

    # Add Performance Metrics row labels
    # 6 Test Performance Metrics groups, 7 metrics each
    num_groups = 6
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'Sensitivity', 'F1-Score', 'AUC']

    for group in range(1, num_groups + 1):
        for metric_name in metrics_names:
            data['Performance Metrics'].append(f'Test Group {group} - {metric_name}')

    print(f"Found {len(txt_files)} TXT files, starting processing...")

    # Process each file
    for filename in txt_files:
        file_path = os.path.join(folder_path, filename)
        col_name = os.path.splitext(filename)[0]

        try:
            print(f"\nProcessing file: {filename}")

            # Extract all metric groups
            metrics_list = extract_all_metrics(file_path)

            print(f"  Actually extracted {len(metrics_list)} groups of metrics")

            # Ensure we have 6 groups of data (fill with None if insufficient)
            while len(metrics_list) < num_groups:
                metrics_list.append((None, None, None, None, None, None, None))
                print(f"  Filling missing data group, current total: {len(metrics_list)}")

            # Only take first 6 groups of data
            metrics_list = metrics_list[:num_groups]

            # Add all 7 metrics from each group to column in order
            for i, (acc, prec, rec, spec, sens, f1, auc) in enumerate(metrics_list):
                data[col_name].extend([acc, prec, rec, spec, sens, f1, auc])
                print(f"  Group {i + 1} metrics added")

            print(f"  ✓ {filename}: Successfully extracted {len(metrics_list)} groups of metrics")
        except Exception as e:
            print(f"  ✗ Error processing file {filename}: {str(e)}")
            # Fill with empty values on error
            for _ in range(num_groups * 7):
                data[col_name].append(None)

    # Create DataFrame
    df = pd.DataFrame(data)

    # Save to Excel
    df.to_excel(output_excel, index=False)

    print(f"\nProcessing completed! Results saved to: {output_excel}")
    print(f"Output format: {len(txt_files)} columns (files) + 1 column (metric names) × {num_groups * 7} rows")
    print(f"Total rows: {len(df)} rows")
    print(f"Metrics per group: {', '.join(metrics_names)}")


if __name__ == "__main__":
    main()
