# In[1]:
# common_utils.py - Common utility functions for the project
# This file contains various utility functions

import os
import sys
import warnings
from scipy.stats import chi2_contingency
warnings.filterwarnings('ignore')
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
import random

# Machine learning imports
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc as sklearn_auc
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# TensorFlow related
import tensorflow as tf
def setup_gpu():
    """GPU"""
    print(tf.test.is_built_with_cuda())
    print(tf.config.list_physical_devices('GPU'))
    # check
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Set GPU memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
        except RuntimeError as e:
            print(e)

# Keras related
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


# In[2]:
# Add special seed constants at the top of the file
FIXED_SPLIT_SEED = 9999  # Special seed 9999, used for fixed development/test set split
DEV_TEST_SEED = 8    # [0,1,2,3,4,5,6,7,8] for random dev/test split; 9999 for fixed split
TRAIN_VAL_SEED = 30  # [100, 300, 500] # Default seed 30, but will be overridden by random selection in gmm.py and cnn.py

# In[3]:
def normalize_mel(x, delta=0, norm_mode='per sample'):
    """
    Normalize mel spectrogram features
    Args:
        x: Input mel spectrogram data
        delta: Order of delta features (0, 1, or 2)
        norm_mode: Normalization mode ('per sample' or 'training data')
    Returns:
        Normalized mel spectrogram data
    """

    mel_log = np.log10(x + 1e-10)

    if delta == 2:
        delta_1 = librosa.feature.delta(mel_log, order=1, mode='nearest')
        delta_2 = librosa.feature.delta(mel_log, order=2, mode='nearest')
        x_log = np.concatenate((mel_log, delta_1, delta_2), axis=3)
    elif delta == 1:
        delta_1 = librosa.feature.delta(mel_log, order=1, mode='nearest')
        x_log = np.concatenate((mel_log, delta_1), axis=3)
    else:
        x_log = mel_log.copy()

    x_norm = x_log.copy()

    if norm_mode == 'per sample':
        for aud in range(x_log.shape[0]):  # each audio
            for f in range(x_log.shape[-1]):
                for freq in range(x_log.shape[1]):
                    mean = np.mean(x_log[aud, freq, :, f])
                    std = np.std(x_log[aud, freq, :, f])
                    x_norm[aud, freq, :, f] = (x_log[aud, freq, :, f] - mean) / std
        return x_norm

    elif norm_mode == 'training data':
        spec_mean = np.zeros((128, x_log.shape[-1]))
        spec_std = np.zeros((128, x_log.shape[-1]))

        for freq in range(x_log.shape[1]):
            for f in range(x_log.shape[-1]):
                spec_mean[freq, f] = np.mean(x_log[:, freq, :, f])
                spec_std[freq, f] = np.std(x_log[:, freq, :, f])
                x_norm[:, freq, :, f] = (x_log[:, freq, :, f] - spec_mean[freq, f]) / spec_std[freq, f]
        return x_norm, spec_mean, spec_std
    else:
        return x_norm

# In[4]:
def analyze_dataset_distribution(patient_ids, segment_ids_a, segment_labels_a,
                                 segment_ids_i, segment_labels_i, age_group_map, gender_map,
                                 split_df, dataset_name="Dataset", verbose=True):
    """
    Analyze dataset distribution, counting segments for vowels a and i separately

    Args:
        patient_ids: List of patient IDs
        segment_ids_a: List of segment IDs for vowel a
        segment_labels_a: List of segment labels for vowel a
        segment_ids_i: List of segment IDs for vowel i
        segment_labels_i: List of segment labels for vowel i
        age_group_map: Age group mapping dictionary (patient_id -> age_group)
        gender_map: Gender mapping dictionary (patient_id -> gender)
        split_df: Complete DataFrame of the data
        dataset_name: Name of the dataset for display purposes
        verbose: Whether to output detailed analysis information, default True

    Returns:
        Dictionary containing comprehensive statistics of the dataset including
        patient-level and segment-level distributions across age, gender, and disease status
    """

    # Simplified mode: return only basic statistics without detailed output
    if not verbose:
        return {
            'patient_stats': {
                'total_patients': len(patient_ids),
            },
            'segment_stats': {
                'total_segments': len(segment_ids_a) + len(segment_ids_i),
            }
        }

    # =================== PATIENT-LEVEL ANALYSIS ===================
    print(f"【PATIENT-LEVEL ANALYSIS】")
    print(f"{dataset_name} Detailed Distribution Analysis:")
    print(f"Total Patients: {len(patient_ids)}")

    # Create DataFrame for this specific dataset
    dataset_df = split_df[split_df['ID'].isin(patient_ids)].copy()

    # Add age group and gender information to the dataset
    dataset_df['Age_Group'] = dataset_df['ID'].map(age_group_map)
    dataset_df['Gender'] = dataset_df['ID'].map(gender_map)

    # Analyze age group distribution at patient level
    print(f"Age Group Distribution:")
    age_groups = [0, 1, 2]
    age_group_names = ['<35 years', '35-50 years', '>50 years']
    patient_age_stats = {}
    for age_group, age_name in zip(age_groups, age_group_names):
        count = len([pid for pid in patient_ids if age_group_map.get(pid, -1) == age_group])
        patient_age_stats[age_group] = count
        print(f"    {age_name} (Group {age_group}): {count} patients")

    # Analyze gender distribution at patient level
    print(f"Gender Distribution:")
    gender_groups = [0, 1]
    gender_names = ['Male', 'Female']
    patient_gender_stats = {}
    for gender_group, gender_name in zip(gender_groups, gender_names):
        count = len([pid for pid in patient_ids if gender_map.get(pid, -1) == gender_group])
        patient_gender_stats[gender_group] = count
        print(f"    {gender_name} (Group {gender_group}): {count} patients")

    # Analyze disease status distribution at patient level
    print(f"Disease Status Distribution:")
    class_counts = dataset_df['Class'].value_counts().sort_index()
    patient_class_stats = {}
    for class_label, count in class_counts.items():
        status = "Diseased" if class_label == 1 else "Healthy"
        patient_class_stats[class_label] = count
        print(f"    {status} (Class {class_label}): {count} patients")

    # Cross-tabulation: Age Group × Disease Status
    print(f"Age Group × Disease Status Cross-analysis:")
    for age_group, age_name in zip(age_groups, age_group_names):
        age_patients = [pid for pid in patient_ids if age_group_map.get(pid, -1) == age_group]
        age_df = dataset_df[dataset_df['ID'].isin(age_patients)]
        if not age_df.empty:
            healthy = len(age_df[age_df['Class'] == 0])
            sick = len(age_df[age_df['Class'] == 1])
            print(f"    {age_name}: Healthy={healthy} patients, Diseased={sick} patients")

    # Cross-tabulation: Gender × Disease Status
    print(f"Gender × Disease Status Cross-analysis:")
    for gender_group, gender_name in zip(gender_groups, gender_names):
        gender_patients = [pid for pid in patient_ids if gender_map.get(pid, -1) == gender_group]
        gender_df = dataset_df[dataset_df['ID'].isin(gender_patients)]
        if not gender_df.empty:
            healthy = len(gender_df[gender_df['Class'] == 0])
            sick = len(gender_df[gender_df['Class'] == 1])
            print(f"    {gender_name}: Healthy={healthy} patients, Diseased={sick} patients")

    # =================== SEGMENT-LEVEL ANALYSIS ===================
    print(f"\n【SEGMENT-LEVEL ANALYSIS】")
    print(f"{dataset_name} Detailed Distribution Analysis:")

    # Combine vowel a and i segment data for overall statistics
    all_segment_ids = segment_ids_a + segment_ids_i
    all_segment_labels = segment_labels_a + segment_labels_i
    print(f"Total Segments: {len(all_segment_ids)} (Vowel a: {len(segment_ids_a)}, Vowel i: {len(segment_ids_i)})")

    # Calculate segments per patient statistics
    segment_counts_per_patient = {}
    for seg_id in all_segment_ids:
        segment_counts_per_patient[seg_id] = segment_counts_per_patient.get(seg_id, 0) + 1

    print(f"Segment Count Statistics:")
    print(f"  Average segments per patient: {len(all_segment_ids) / len(patient_ids):.1f}")
    print(f"  Minimum segments per patient: {min(segment_counts_per_patient.values())}")
    print(f"  Maximum segments per patient: {max(segment_counts_per_patient.values())}")

    # =================== VOWEL A SEGMENT ANALYSIS ===================
    print(f"\n【VOWEL A SEGMENT ANALYSIS】")
    print(f"Total Vowel A Segments: {len(segment_ids_a)}")

    # Ensure segment_labels_a is a numpy array for efficient processing
    segment_labels_a_array = np.array(segment_labels_a).flatten()

    # Vowel a segments by age group
    print(f"Vowel A Age Group Distribution:")
    segment_a_age_stats = {}
    for age_group, age_name in zip(age_groups, age_group_names):
        age_patients = [pid for pid in patient_ids if age_group_map.get(pid, -1) == age_group]
        segment_count = sum(1 for seg_id in segment_ids_a if seg_id in age_patients)
        segment_a_age_stats[age_group] = segment_count
        print(f"    {age_name} (Group {age_group}): {segment_count} segments")

    # Vowel a segments by gender
    print(f"Vowel A Gender Distribution:")
    segment_a_gender_stats = {}
    for gender_group, gender_name in zip(gender_groups, gender_names):
        gender_patients = [pid for pid in patient_ids if gender_map.get(pid, -1) == gender_group]
        segment_count = sum(1 for seg_id in segment_ids_a if seg_id in gender_patients)
        segment_a_gender_stats[gender_group] = segment_count
        print(f"    {gender_name} (Group {gender_group}): {segment_count} segments")

    # Vowel a segments by disease status
    print(f"Vowel A Disease Status Distribution:")
    segment_a_class_stats = {}
    for class_label in [0, 1]:
        status = "Diseased" if class_label == 1 else "Healthy"
        segment_count = np.sum(segment_labels_a_array == class_label)
        segment_a_class_stats[class_label] = segment_count
        print(f"    {status} (Class {class_label}): {segment_count} segments")

    # Cross-tabulation: Vowel a Age Group × Disease Status
    print(f"Vowel A Age Group × Disease Status:")
    for age_group, age_name in zip(age_groups, age_group_names):
        age_patients = [pid for pid in patient_ids if age_group_map.get(pid, -1) == age_group]
        age_segment_indices = [i for i, seg_id in enumerate(segment_ids_a) if seg_id in age_patients]
        if age_segment_indices:
            age_segment_labels = segment_labels_a_array[age_segment_indices]
            healthy_segments = np.sum(age_segment_labels == 0)
            sick_segments = np.sum(age_segment_labels == 1)
            print(f"    {age_name}: Healthy={healthy_segments} segments, Diseased={sick_segments} segments")

    # Cross-tabulation: Vowel a Gender × Disease Status
    print(f"Vowel A Gender × Disease Status:")
    for gender_group, gender_name in zip(gender_groups, gender_names):
        gender_patients = [pid for pid in patient_ids if gender_map.get(pid, -1) == gender_group]
        gender_segment_indices = [i for i, seg_id in enumerate(segment_ids_a) if seg_id in gender_patients]
        if gender_segment_indices:
            gender_segment_labels = segment_labels_a_array[gender_segment_indices]
            healthy_segments = np.sum(gender_segment_labels == 0)
            sick_segments = np.sum(gender_segment_labels == 1)
            print(f"    {gender_name}: Healthy={healthy_segments} segments, Diseased={sick_segments} segments")

    # =================== VOWEL I SEGMENT ANALYSIS ===================
    print(f"\n【VOWEL I SEGMENT ANALYSIS】")
    print(f"Total Vowel I Segments: {len(segment_ids_i)}")

    # Ensure segment_labels_i is a numpy array for efficient processing
    segment_labels_i_array = np.array(segment_labels_i).flatten()

    # Vowel i segments by age group
    print(f"Vowel I Age Group Distribution:")
    segment_i_age_stats = {}
    for age_group, age_name in zip(age_groups, age_group_names):
        age_patients = [pid for pid in patient_ids if age_group_map.get(pid, -1) == age_group]
        segment_count = sum(1 for seg_id in segment_ids_i if seg_id in age_patients)
        segment_i_age_stats[age_group] = segment_count
        print(f"    {age_name} (Group {age_group}): {segment_count} segments")

    # Vowel i segments by gender
    print(f"Vowel I Gender Distribution:")
    segment_i_gender_stats = {}
    for gender_group, gender_name in zip(gender_groups, gender_names):
        gender_patients = [pid for pid in patient_ids if gender_map.get(pid, -1) == gender_group]
        segment_count = sum(1 for seg_id in segment_ids_i if seg_id in gender_patients)
        segment_i_gender_stats[gender_group] = segment_count
        print(f"    {gender_name} (Group {gender_group}): {segment_count} segments")

    # Vowel i segments by disease status
    print(f"Vowel I Disease Status Distribution:")
    segment_i_class_stats = {}
    for class_label in [0, 1]:
        status = "Diseased" if class_label == 1 else "Healthy"
        segment_count = np.sum(segment_labels_i_array == class_label)
        segment_i_class_stats[class_label] = segment_count
        print(f"    {status} (Class {class_label}): {segment_count} segments")

    # Cross-tabulation: Vowel i Age Group × Disease Status
    print(f"Vowel I Age Group × Disease Status:")
    for age_group, age_name in zip(age_groups, age_group_names):
        age_patients = [pid for pid in patient_ids if age_group_map.get(pid, -1) == age_group]
        age_segment_indices = [i for i, seg_id in enumerate(segment_ids_i) if seg_id in age_patients]
        if age_segment_indices:
            age_segment_labels = segment_labels_i_array[age_segment_indices]
            healthy_segments = np.sum(age_segment_labels == 0)
            sick_segments = np.sum(age_segment_labels == 1)
            print(f"    {age_name}: Healthy={healthy_segments} segments, Diseased={sick_segments} segments")

    # Cross-tabulation: Vowel i Gender × Disease Status
    print(f"Vowel I Gender × Disease Status:")
    for gender_group, gender_name in zip(gender_groups, gender_names):
        gender_patients = [pid for pid in patient_ids if gender_map.get(pid, -1) == gender_group]
        gender_segment_indices = [i for i, seg_id in enumerate(segment_ids_i) if seg_id in gender_patients]
        if gender_segment_indices:
            gender_segment_labels = segment_labels_i_array[gender_segment_indices]
            healthy_segments = np.sum(gender_segment_labels == 0)
            sick_segments = np.sum(gender_segment_labels == 1)
            print(f"    {gender_name}: Healthy={healthy_segments} segments, Diseased={sick_segments} segments")

    # Return comprehensive statistics dictionary after all analyses are complete
    return {
        'patient_stats': {
            'total_patients': len(patient_ids),
            'age_distribution': patient_age_stats,
            'gender_distribution': patient_gender_stats,
            'class_distribution': patient_class_stats
        },
        'segment_stats': {
            'total_segments': len(all_segment_ids),
            'segments_per_patient': segment_counts_per_patient,
            'age_distribution': {
                'total': {age_group: sum(1 for seg_id in all_segment_ids
                                         if age_group_map.get(seg_id, -1) == age_group)
                          for age_group in age_groups},
                'vowel_a': segment_a_age_stats,
                'vowel_i': segment_i_age_stats
            },
            'gender_distribution': {
                'total': {gender_group: sum(1 for seg_id in all_segment_ids
                                            if gender_map.get(seg_id, -1) == gender_group)
                          for gender_group in gender_groups},
                'vowel_a': segment_a_gender_stats,
                'vowel_i': segment_i_gender_stats
            },
            'class_distribution': {
                'total': {0: np.sum(np.array(all_segment_labels) == 0),
                          1: np.sum(np.array(all_segment_labels) == 1)},
                'vowel_a': segment_a_class_stats,
                'vowel_i': segment_i_class_stats
            }
        }
    }
# In[5]:
def create_composite_stratify_key(data_df, age_group_map, gender_map):
    """
    Create composite stratification key for multi-dimensional stratified sampling

    Args:
        data_df: DataFrame containing patient ID and Class columns
        age_group_map: Age group mapping {patient_id: age_group}
        gender_map: Gender mapping {patient_id: gender}

    Returns:
        composite_key: List of composite stratification keys
    """
    composite_key = []

    for _, row in data_df.iterrows():
        patient_id = row['ID']
        disease_class = row['Class']
        age_group = age_group_map.get(patient_id, -1)
        gender = gender_map.get(patient_id, -1)

        # Create composite key: "age_group_gender_disease_status"
        composite = f"{age_group}_{gender}_{disease_class}"
        composite_key.append(composite)

    return composite_key


def stratified_patient_split(split_df, age_group_map, gender_map, test_size=0.2, random_state=0, verbose=True):
    """
    Multi-dimensional stratified patient split based on age, gender, and disease status

    Args:
        split_df: Patient data DataFrame containing ID and Class columns
        age_group_map: Age group mapping dictionary
        gender_map: Gender mapping dictionary
        test_size: Test set proportion
        random_state: Random seed for reproducibility
        verbose: Whether to output detailed stratification information, default True

    Returns:
        dev_ids, test_ids: Lists of patient IDs for development and test sets
    """

    if verbose:
        print("=" * 60)
        print("Multi-dimensional Stratified Patient Split")
        print("=" * 60)

    # Create composite stratification key
    composite_stratify = create_composite_stratify_key(split_df, age_group_map, gender_map)

    # Count samples in each combination
    unique_combinations, counts = np.unique(composite_stratify, return_counts=True)
    if verbose:
        print("Stratification Combination Statistics:")
        for combo, count in zip(unique_combinations, counts):
            age, gender, disease = combo.split('_')
            age_name = {0: '<35 years', 1: '35-50 years', 2: '>50 years'}.get(int(age), f'Unknown({age})')
            gender_name = {0: 'Male', 1: 'Female'}.get(int(gender), f'Unknown({gender})')
            disease_name = {0: 'Healthy', 1: 'Diseased'}.get(int(disease), f'Unknown({disease})')
            print(f"  {age_name}+{gender_name}+{disease_name}: {count} patients")

    # Check for combinations with insufficient samples
    min_samples = 2  # Each group needs at least 2 samples for stratification
    small_groups = [(combo, count) for combo, count in zip(unique_combinations, counts) if count < min_samples]

    if small_groups:
        if verbose:
            print(f"\nWarning: Found {len(small_groups)} combinations with fewer than {min_samples} samples:")
            for combo, count in small_groups:
                print(f"  {combo}: {count} patients")
            print("Falling back to disease status only stratification...")

        # Fallback to single stratification
        dev_ids, test_ids = train_test_split(
            split_df['ID'],
            test_size=test_size,
            stratify=split_df['Class'],
            random_state=random_state
        )
    else:
        # Perform multi-dimensional stratified split
        dev_ids, test_ids = train_test_split(
            split_df['ID'],
            test_size=test_size,
            stratify=composite_stratify,
            random_state=random_state
        )
        print(f"Successfully completed multi-dimensional stratified split")

    # Validate split results
    if verbose:
        print(f"\nSplit Result Validation:")
        print(f"Development set patients: {len(dev_ids)} ({len(dev_ids) / len(split_df) * 100:.1f}%)")
        print(f"Test set patients: {len(test_ids)} ({len(test_ids) / len(split_df) * 100:.1f}%)")

        # Detailed proportion verification
        verify_patient_split_proportions(split_df, dev_ids, test_ids, age_group_map, gender_map)

    return dev_ids.tolist(), test_ids.tolist()


def create_segment_composite_stratify_key(segment_ids, segment_labels, age_group_map, gender_map):
    """
    Create composite stratification key for segment-level data

    Args:
        segment_ids: List of patient IDs corresponding to segments
        segment_labels: Segment labels
        age_group_map: Age group mapping dictionary
        gender_map: Gender mapping dictionary

    Returns:
        composite_key: List of composite stratification keys for segments
    """
    composite_key = []

    for i, patient_id in enumerate(segment_ids):
        # Handle different label formats (scalar, array, nested array)
        label = segment_labels[i] if hasattr(segment_labels[i], '__len__') else segment_labels[i]
        if hasattr(label, '__len__') and len(label) > 0:
            label = label[0] if hasattr(label[0], '__len__') else label[0]

        age_group = age_group_map.get(patient_id, -1)
        gender = gender_map.get(patient_id, -1)

        # Create composite key: "age_group_gender_disease_status"
        composite = f"{age_group}_{gender}_{int(label)}"
        composite_key.append(composite)

    return composite_key


def stratified_segment_split(x_data, y_data, id_data, age_group_map, gender_map,
                             test_size=0.2, random_state=300, vowel_type="unknown", verbose=True):
    """
    Multi-dimensional stratified segment split based on age, gender, and disease status

    Args:
        x_data: Feature data
        y_data: Label data
        id_data: Patient ID data
        age_group_map: Age group mapping dictionary
        gender_map: Gender mapping dictionary
        test_size: Validation set proportion
        random_state: Random seed for reproducibility
        vowel_type: Vowel type identifier for logging
        verbose: Whether to output detailed stratification information, default True

    Returns:
        x_train, x_val, y_train, y_val, id_train, id_val: Split training and validation data
    """
    print(f"🎯 stratified_segment_split actual seed used: {random_state}")
    if verbose:
        print(f"\n{'=' * 40}")
        print(f"Vowel {vowel_type} Multi-dimensional Stratified Segment Split")
        print(f"{'=' * 40}")

    # Create composite stratification key
    composite_stratify = create_segment_composite_stratify_key(id_data, y_data, age_group_map, gender_map)

    # Count samples in each combination
    unique_combinations, counts = np.unique(composite_stratify, return_counts=True)
    if verbose:
        print(f"Segment Stratification Combination Statistics:")
        for combo, count in zip(unique_combinations, counts):
            age, gender, disease = combo.split('_')
            age_name = {0: '<35 years', 1: '35-50 years', 2: '>50 years'}.get(int(age), f'Unknown({age})')
            gender_name = {0: 'Male', 1: 'Female'}.get(int(gender), f'Unknown({gender})')
            disease_name = {0: 'Healthy', 1: 'Diseased'}.get(int(disease), f'Unknown({disease})')
            print(f"  {age_name}+{gender_name}+{disease_name}: {count} segments")

    # Check for insufficient samples in any combination
    min_samples = 2
    small_groups = [(combo, count) for combo, count in zip(unique_combinations, counts) if count < min_samples]

    if small_groups:
        if verbose:
            print(
                f"\nWarning: Vowel {vowel_type} found {len(small_groups)} combinations with insufficient samples, falling back to disease status stratification...")

        # Fallback to single stratification based on disease status only
        x_train, x_val, y_train, y_val, id_train, id_val = train_test_split(
            x_data, y_data, id_data,
            test_size=test_size,
            random_state=random_state,
            stratify=y_data
        )
    else:
        # Perform multi-dimensional stratified split
        x_train, x_val, y_train, y_val, id_train, id_val = train_test_split(
            x_data, y_data, id_data,
            test_size=test_size,
            random_state=random_state,
            stratify=composite_stratify
        )
        if verbose:
            print(f"Vowel {vowel_type} successfully completed multi-dimensional stratified split")

    # Validate split results
    if verbose:
        print(f"Split Results:")
        print(f"  Training set: {len(x_train)} segments")
        print(f"  Validation set: {len(x_val)} segments")

    return x_train, x_val, y_train, y_val, id_train, id_val


def verify_patient_split_proportions(original_df, dev_ids, test_ids, age_group_map, gender_map):
    """
    Verify that proportions are maintained after patient split

    Args:
        original_df: Original DataFrame with all patients
        dev_ids: Development set patient IDs
        test_ids: Test set patient IDs
        age_group_map: Age group mapping dictionary
        gender_map: Gender mapping dictionary
    """

    def calculate_proportions(patient_ids, name):
        """Calculate and display proportions for a given set of patient IDs"""
        subset_df = original_df[original_df['ID'].isin(patient_ids)]
        total = len(patient_ids)

        print(f"\n{name} Proportion Analysis:")

        # Disease status proportions
        disease_counts = subset_df['Class'].value_counts().sort_index()
        for disease, count in disease_counts.items():
            disease_name = "Healthy" if disease == 0 else "Diseased"
            print(f"  {disease_name}: {count} patients ({count / total * 100:.1f}%)")

        # Age group proportions
        age_counts = {}
        for pid in patient_ids:
            age_group = age_group_map.get(pid, -1)
            age_counts[age_group] = age_counts.get(age_group, 0) + 1

        for age_group in sorted(age_counts.keys()):
            age_name = {0: '<35 years', 1: '35-50 years', 2: '>50 years'}.get(age_group, f'Unknown({age_group})')
            count = age_counts[age_group]
            print(f"  {age_name}: {count} patients ({count / total * 100:.1f}%)")

        # Gender proportions
        gender_counts = {}
        for pid in patient_ids:
            gender = gender_map.get(pid, -1)
            gender_counts[gender] = gender_counts.get(gender, 0) + 1

        for gender in sorted(gender_counts.keys()):
            gender_name = {0: 'Male', 1: 'Female'}.get(gender, f'Unknown({gender})')
            count = gender_counts[gender]
            print(f"  {gender_name}: {count} patients ({count / total * 100:.1f}%)")

    # Calculate proportions for each dataset split
    all_ids = original_df['ID'].tolist()
    calculate_proportions(all_ids, "Original Data")
    calculate_proportions(dev_ids, "Development Set")
    calculate_proportions(test_ids, "Test Set")


# In[6]:
def get_data_pkl_both_a_i(base_folder, ftype, test_size=0.2, verbose=True):
    """
    Load and split audio feature data for both vowels 'a' and 'i' into development and test sets.

    Args:
        base_folder: Base directory containing the pickle files
        ftype: Feature type (e.g., 'mel' for mel spectrograms)
        test_size: Proportion of data for test set (default 0.2)
        verbose: Whether to print detailed analysis (default True)

    Returns:
        Tuple containing train/val/test splits for both vowels and sensitive attributes
    """

    # Load patient metadata (including age and gender)
    age_group_map, gender_map = load_patient_metadata('chinese')

    # Load raw speech feature data
    read_path_a = os.path.join(base_folder, 'vowel-a_{}_ch.pkl'.format(ftype))
    read_path_i = os.path.join(base_folder, 'vowel-i_{}_ch.pkl'.format(ftype))
    dataset_a = joblib.load(read_path_a)
    features_raw_a, labels_raw_a, ids_raw_a = dataset_a['features'], dataset_a['labels'], dataset_a['ids']
    dataset_i = joblib.load(read_path_i)
    features_raw_i, labels_raw_i, ids_raw_i = dataset_i['features'], dataset_i['labels'], dataset_i['ids']

    split_df = pd.read_csv('D:/data/audio/EENT/EENT_sum.csv', encoding='utf-8-sig')

    # Original complete data distribution (only shown in verbose mode)
    if verbose:
        print(f"\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Complete Dataset Detailed Analysis:")
        all_patient_ids = split_df['ID'].tolist()
        all_segment_ids_a = ids_raw_a
        all_segment_ids_i = ids_raw_i
        all_segment_labels_a = [1 if label != 0 else 0 for label in labels_raw_a]
        all_segment_labels_i = [1 if label != 0 else 0 for label in labels_raw_i]
        analyze_dataset_distribution(all_patient_ids, all_segment_ids_a, all_segment_labels_a,
                                     all_segment_ids_i, all_segment_labels_i, age_group_map, gender_map,
                                     split_df, "Original Complete Dataset", verbose=verbose)

    # ==================2.2 Fixed Patient Split ==========================
    if DEV_TEST_SEED == FIXED_SPLIT_SEED:
        print(f"\n{'=' * 60}")
        print("Using fixed development/test set split")
        print(f"{'=' * 60}")

        # Use fixed patient split
        fixed_split_df = pd.read_csv('D:/data/audio/EENT/Subject selection and allocation.csv', encoding='utf-8-sig')
        pt_id_develop = list(fixed_split_df['Development'])
        pt_id_test = list(fixed_split_df['Test'].dropna())
        print(f"Loaded from fixed split file:")
        print(f"  Development set patients: {len(pt_id_develop)}")
        print(f"  Test set patients: {len(pt_id_test)}")

        # Validate that fixed split patients exist in the dataset
        all_patient_ids = set(split_df['ID'].tolist())
        missing_dev = [pid for pid in pt_id_develop if pid not in all_patient_ids]
        missing_test = [pid for pid in pt_id_test if pid not in all_patient_ids]
        if missing_dev:
            print(f"  Warning: {len(missing_dev)} development patients not in current dataset")
            print(f"      Missing patient IDs: {missing_dev[:5]}{'...' if len(missing_dev) > 5 else ''}")
            # Filter out non-existent patients
            pt_id_develop = [pid for pid in pt_id_develop if pid in all_patient_ids]
        if missing_test:
            print(f"  Warning: {len(missing_test)} test patients not in current dataset")
            print(f"      Missing patient IDs: {missing_test[:5]}{'...' if len(missing_test) > 5 else ''}")
            # Filter out non-existent patients
            pt_id_test = [pid for pid in pt_id_test if pid in all_patient_ids]
        print(f"  Actually used:")
        print(f"    Development set patients: {len(pt_id_develop)}")
        print(f"    Test set patients: {len(pt_id_test)}")

        # Check coverage rate
        total_patients = len(all_patient_ids)
        covered_patients = len(set(pt_id_develop) | set(pt_id_test))
        coverage_rate = covered_patients / total_patients
        print(f"  Coverage rate: {covered_patients}/{total_patients} = {coverage_rate:.1%}")
        if coverage_rate < 0.95:  # Warning if coverage rate is below 95%
            print(f"  Warning: Low patient coverage rate ({coverage_rate:.1%})")

    # ==================2.2 Multi-dimensional Stratified Development and Test Patient Split ==========================
    else:
        # Use original multi-dimensional stratified patient split
        if verbose:
            print(f"\n{'=' * 60}")
            print("Starting multi-dimensional stratified development/test split")
            print(f"{'=' * 60}")
        pt_id_develop, pt_id_test = stratified_patient_split(
            split_df, age_group_map, gender_map,
            test_size=0.2, random_state=DEV_TEST_SEED, verbose=verbose
        )
    # =========================================================================

    if verbose:
        print(f"Random seed used: development/test={DEV_TEST_SEED}")
        print(f"Random seed used: train/validation={TRAIN_VAL_SEED}")

    # Initialize development set variables - collect development set segments
    feature_develop_a = []
    label_develop_a = []
    ids_develop_a = []
    feature_develop_i = []
    label_develop_i = []
    ids_develop_i = []

    # Collect development set data for each patient
    for pt_id in pt_id_develop:
        # Collect vowel 'a' data
        indices_a = [i for i, tar in enumerate(ids_raw_a) if tar == pt_id]
        for idx in indices_a:
            feature_develop_a.append(features_raw_a[idx])
            label_develop_a.append(labels_raw_a[idx])
            ids_develop_a.append(ids_raw_a[idx])

        # Collect vowel 'i' data
        indices_i = [i for i, tar in enumerate(ids_raw_i) if tar == pt_id]
        for idx in indices_i:
            feature_develop_i.append(features_raw_i[idx])
            label_develop_i.append(labels_raw_i[idx])
            ids_develop_i.append(ids_raw_i[idx])

    # Development set data distribution (only shown in verbose mode)
    if verbose:
        print(f"\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Development Set Detailed Analysis:")
        develop_segment_labels_a = [1 if label != 0 else 0 for label in label_develop_a]
        develop_segment_labels_i = [1 if label != 0 else 0 for label in label_develop_i]
        analyze_dataset_distribution(pt_id_develop, ids_develop_a, develop_segment_labels_a,
                                     ids_develop_i, develop_segment_labels_i, age_group_map, gender_map,
                                     split_df, "Development Set", verbose=verbose)

    # Convert to numpy arrays
    x_develop_a = np.vstack(feature_develop_a)
    y_develop_a = np.vstack(label_develop_a)
    y_develop_a = np.where(np.equal(y_develop_a, 0), 0, 1)
    x_develop_i = np.vstack(feature_develop_i)
    y_develop_i = np.vstack(label_develop_i)
    y_develop_i = np.where(np.equal(y_develop_i, 0), 0, 1)

    if verbose:
        # Random split for training and validation sets
        print(f"Random split within development set (train_val_seed={TRAIN_VAL_SEED}):")
        print(f"  Vowel a: {x_develop_a.shape[0]} samples, Vowel i: {x_develop_i.shape[0]} samples")

    # ================2.2 Multi-dimensional Stratified Training and Validation Segment Split ==============================
    # Training and validation set split
    if verbose:
        print(f"\n{'=' * 60}")
        print("Starting multi-dimensional stratified train/validation split")
        print(f"{'=' * 60}")
        print(f"Random split within development set (train_val_seed={TRAIN_VAL_SEED}):")
        print(f"  Vowel a: {x_develop_a.shape[0]} samples, Vowel i: {x_develop_i.shape[0]} samples")

    # Multi-dimensional stratified split for vowel 'a'
    x_train_a, x_val_a, y_train_a, y_val_a, id_train_a, id_val_a = stratified_segment_split(
        x_develop_a, y_develop_a, ids_develop_a, age_group_map, gender_map,
        test_size=0.2, random_state=TRAIN_VAL_SEED, vowel_type="a", verbose=verbose)

    # Multi-dimensional stratified split for vowel 'i'
    x_train_i, x_val_i, y_train_i, y_val_i, id_train_i, id_val_i = stratified_segment_split(
        x_develop_i, y_develop_i, ids_develop_i, age_group_map, gender_map,
        test_size=0.2, random_state=TRAIN_VAL_SEED, vowel_type="i", verbose=verbose)
    # ===============================================================================================

    # Training and validation set data analysis (only shown in verbose mode)
    if verbose:
        print(f"\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Training Set Detailed Analysis:")
        train_patient_ids = list(set(id_train_a + id_train_i))
        train_segment_labels_a_binary = [1 if label != 0 else 0 for label in y_train_a.flatten()]
        train_segment_labels_i_binary = [1 if label != 0 else 0 for label in y_train_i.flatten()]
        analyze_dataset_distribution(train_patient_ids, id_train_a, train_segment_labels_a_binary,
                                     id_train_i, train_segment_labels_i_binary, age_group_map, gender_map,
                                     split_df, "Training Set", verbose=verbose)

        print(f"\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Validation Set Detailed Analysis:")
        val_patient_ids = list(set(id_val_a + id_val_i))
        val_segment_labels_a_binary = [1 if label != 0 else 0 for label in y_val_a.flatten()]
        val_segment_labels_i_binary = [1 if label != 0 else 0 for label in y_val_i.flatten()]
        analyze_dataset_distribution(val_patient_ids, id_val_a, val_segment_labels_a_binary,
                                     id_val_i, val_segment_labels_i_binary, age_group_map, gender_map,
                                     split_df, "Validation Set", verbose=verbose)

    # Build test set
    feature_test_a = []
    label_test_a = []
    id_test_a = []
    feature_test_i = []
    label_test_i = []
    id_test_i = []

    # Collect test set data for each patient
    for pt_id in pt_id_test:
        # Collect vowel 'a' test data
        indices_a = [i for i, tar in enumerate(ids_raw_a) if tar == pt_id]
        for idx in indices_a:
            feature_test_a.append(features_raw_a[idx])
            label_test_a.append(labels_raw_a[idx])
            id_test_a.append(ids_raw_a[idx])

        # Collect vowel 'i' test data
        indices_i = [i for i, tar in enumerate(ids_raw_i) if tar == pt_id]
        for idx in indices_i:
            feature_test_i.append(features_raw_i[idx])
            label_test_i.append(labels_raw_i[idx])
            id_test_i.append(ids_raw_i[idx])

    # Test set data analysis (only shown in verbose mode)
    if verbose:
        print(f"\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Test Set Detailed Analysis:")
        test_segment_labels_a_binary = [1 if label != 0 else 0 for label in label_test_a]
        test_segment_labels_i_binary = [1 if label != 0 else 0 for label in label_test_i]
        analyze_dataset_distribution(pt_id_test, id_test_a, test_segment_labels_a_binary,
                                     id_test_i, test_segment_labels_i_binary, age_group_map, gender_map,
                                     split_df, "Test Set", verbose=verbose)

    # Convert test set to numpy arrays
    y_test_a = np.vstack(label_test_a)
    y_test_a = np.where(np.equal(y_test_a, 0), 0, 1)
    y_test_i = np.vstack(label_test_i)
    y_test_i = np.where(np.equal(y_test_i, 0), 0, 1)
    x_test_a = np.vstack(feature_test_a)
    x_test_i = np.vstack(feature_test_i)

    # If mel spectrograms, perform normalization processing
    if ftype == 'mel':
        if verbose:
            print("Preprocessing mel spectrograms...")
        # Extract only the first channel for mel spectrograms
        x_train_a, x_val_a, x_test_a = x_train_a[:, :, :, [0]], x_val_a[:, :, :, [0]], x_test_a[:, :, :, [0]]
        x_train_i, x_val_i, x_test_i = x_train_i[:, :, :, [0]], x_val_i[:, :, :, [0]], x_test_i[:, :, :, [0]]

        # Normalization processing
        x_train_a = normalize_mel(x_train_a, delta=0, norm_mode='per sample')
        x_val_a = normalize_mel(x_val_a, delta=0, norm_mode='per sample')
        x_test_a = normalize_mel(x_test_a, delta=0, norm_mode='per sample')
        x_train_i = normalize_mel(x_train_i, delta=0, norm_mode='per sample')
        x_val_i = normalize_mel(x_val_i, delta=0, norm_mode='per sample')
        x_test_i = normalize_mel(x_test_i, delta=0, norm_mode='per sample')

    # Build sensitive attribute mappings
    train_patient_ids = list(set(id_train_a + id_train_i))
    sensitive_attrs_train = build_sensitive_attrs_dict(train_patient_ids, age_group_map, gender_map)

    val_patient_ids = list(set(id_val_a + id_val_i))
    sensitive_attrs_val = build_sensitive_attrs_dict(val_patient_ids, age_group_map, gender_map)

    test_patient_ids = list(set(id_test_a + id_test_i))
    sensitive_attrs_test = build_sensitive_attrs_dict(test_patient_ids, age_group_map, gender_map)

    return (x_train_a, x_val_a, x_test_a, y_train_a, y_val_a, y_test_a,
            id_train_a, id_val_a, id_test_a, x_train_i, x_val_i, x_test_i,
            y_train_i, y_val_i, y_test_i, id_train_i, id_val_i, id_test_i,
            sensitive_attrs_train, sensitive_attrs_val, sensitive_attrs_test)


# In[7]:
def get_data_pkl_both_a_i_ger(base_folder, feature_type):
    """
    Load German audio feature data for both vowels 'a' and 'i'.

    Args:
        base_folder: Base directory containing the pickle files
        feature_type: Type of features to load (e.g., 'mel')

    Returns:
        Tuple containing German dataset features, labels, IDs and sensitive attributes
    """

    # Load German patient metadata
    age_group_map, gender_map = load_patient_metadata('german')

    # Extract patient IDs and labels
    df = pd.read_csv('D:/data/audio/SVD/German patient labels.csv', encoding='utf-8-sig')
    picked_IDs = set(df['ID'])
    df = df.set_index('ID')

    # Load German vowel 'a' data
    read_path_a = os.path.join(base_folder, 'a_{}_ger.pkl'.format(feature_type))
    dataset_a = joblib.load(read_path_a)
    features_raw_a, labels_raw_a, ids_raw_a = dataset_a['features'], dataset_a['labels'], dataset_a['ids']
    x_ger_a = []
    y_ger_a = []
    id_ger_a = []

    # Filter and collect vowel 'a' data for selected patients
    for i, pid in enumerate(dataset_a['ids']):
        pid = int(pid)
        if pid in picked_IDs:
            x_ger_a.append(dataset_a['features'][i])
            y_ger_a.append(df.loc[pid, 'Class'])
            id_ger_a.append(pid)

    x_ger_a = np.vstack(x_ger_a)
    y_ger_a = np.array(y_ger_a, ndmin=2).T

    # Load German vowel 'i' data
    read_path_i = os.path.join(base_folder, 'i_{}_ger.pkl'.format(feature_type))
    dataset_i = joblib.load(read_path_i)
    features_raw_i, labels_raw_i, ids_raw_i = dataset_i['features'], dataset_i['labels'], dataset_i['ids']
    x_ger_i = []
    y_ger_i = []
    id_ger_i = []

    # Filter and collect vowel 'i' data for selected patients
    for i, pid in enumerate(dataset_i['ids']):
        pid = int(pid)
        if pid in picked_IDs:
            x_ger_i.append(dataset_i['features'][i])
            y_ger_i.append(df.loc[pid, 'Class'])
            id_ger_i.append(pid)

    x_ger_i = np.vstack(x_ger_i)
    y_ger_i = np.array(y_ger_i, ndmin=2).T

    # Process mel spectrograms if feature type is 'mel'
    if feature_type == 'mel':
        # Extract only the first channel and normalize
        x_ger_a, x_ger_i = x_ger_a[:, :, :, [0]], x_ger_i[:, :, :, [0]]
        x_ger_a = normalize_mel(x_ger_a, delta=0, norm_mode='per sample')
        x_ger_i = normalize_mel(x_ger_i, delta=0, norm_mode='per sample')

    # Build sensitive attribute mappings for German dataset
    ger_patient_ids = list(set(id_ger_a + id_ger_i))
    sensitive_attrs_ger = build_sensitive_attrs_dict(ger_patient_ids, age_group_map, gender_map)

    return x_ger_a, y_ger_a, id_ger_a, x_ger_i, y_ger_i, id_ger_i, sensitive_attrs_ger


# In[8]:
def dropna_in_pitches(x):
    """
    Check which rows in the input array do not contain NaN values.
    Args:
        x: Input array to check for NaN values
    Returns:
        List of boolean values indicating whether each row is free of NaN values
    """
    Row_is_not_null = []
    for row in range(len(x)):
        Row_is_not_null.append(True)
        for ele in x[row]:
            if np.isnan(ele):
                Row_is_not_null[-1] = False
                continue

    return Row_is_not_null


# In[9]:
def calculate_performance_metrics(y_true, y_pred):
    """
    Calculate comprehensive performance metrics including confusion matrix components.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Dictionary containing all performance metrics
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        'confusion_matrix': cm,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'sensitivity': sensitivity,
        'f1_score': f1,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }

def print_performance_metrics(metrics, auc_score=None, title="Performance Metrics"):
    """
    Print performance metrics in a formatted way.

    Args:
        metrics: Dictionary containing performance metrics
        auc_score: Optional AUC score to include
        title: Title for the metrics display
    """
    print(f"\n{'=' * 50}")

    # Check if title contains "Val" and modify accordingly for validation sets
    if "Val" in title or "validation" in title.lower():
        modified_title = title.replace("Test Performance Metrics", "Val Performance Metrics")
    else:
        modified_title = title
    print(f"{modified_title}")

    print(f"{'=' * 50}")
    print(metrics['confusion_matrix'])
    print()

    # Check if title contains "Val" and modify metric header accordingly
    if "Val" in title or "validation" in title.lower():
        print(f"Val Performance Metrics:")
    else:
        print(f"Test Performance Metrics:")

    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Precision:   {metrics['precision']:.4f}")
    print(f"  Recall:      {metrics['recall']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"  F1-Score:    {metrics['f1_score']:.4f}")
    if auc_score is not None:
        print(f"  AUC:         {auc_score:.4f}")

    print(f"\nDetailed Counts:")
    print(f"  True Positives:  {metrics['tp']}")
    print(f"  True Negatives:  {metrics['tn']}")
    print(f"  False Positives: {metrics['fp']}")
    print(f"  False Negatives: {metrics['fn']}")
    print(f"{'=' * 50}")


# In[10]:
def compute_fairness_metrics(confusion_matrices_dict):
    """
    Compute fairness metrics from confusion matrices of different demographic groups.

    Input: confusion matrices dict for a demographic group (e.g., by age or gender),
           where each subgroup contains TP, TN, FP, FN values.
    Output: fairness evaluation metrics including:results section
    Args:
        confusion_matrices_dict: Dictionary with group names as keys and confusion matrix components as values

    Returns:
        Dictionary containing various fairness metrics and statistical tests
    """
    groups = list(confusion_matrices_dict.keys())
    n = len(groups)

    # Define epsilon value to replace zeros
    EPSILON = 1e-8  # Small enough not to affect actual calculation results

    results = {
        'accuracy_chi2_p': None,
        'fpr_chi2_p': None,
        'fnr_chi2_p': None,
        'f1_chi2_p': None,
        'AOD': None,
        'DI': {},
        'overall_DI': None,
        'EOP': {},
        'overall_EOP': None,
        'EOD': {},
        'overall_EOD': None,
        'fpr_by_group': {},
        'fnr_by_group': {},
        'f1_by_group': {}
    }

    # ------------------- 1. Chi-square Tests -------------------
    # ------------------- 1.1. Global Accuracy Chi-square Test -------------------
    table_acc = []
    for g in groups:
        cm = confusion_matrices_dict[g]
        correct = cm['TP'] + cm['TN']
        incorrect = cm['FP'] + cm['FN']

        # If incorrect is 0, replace with epsilon
        if incorrect == 0:
            incorrect = EPSILON
            print(
                f"\nNote for accuracy_p calculation: {g} group has 0 incorrect predictions, replaced with epsilon {EPSILON} for chi-square test")

        table_acc.append([correct, incorrect])
    if len(table_acc) >= 2:
        try:
            from scipy.stats import chi2_contingency
            _, p_acc, _, _ = chi2_contingency(table_acc, correction=False)
            results['accuracy_chi2_p'] = p_acc
        except Exception as e:
            print(f"Accuracy chi-square test failed: {str(e)}")
            results['accuracy_chi2_p'] = None

    # ------------------- 1.2. FPR Chi-square Test -------------------
    table_fpr = []
    for g in groups:
        cm = confusion_matrices_dict[g]
        fp = cm['FP']
        tn = cm['TN']

        # If FP or TN is 0, replace with epsilon
        if fp == 0:
            fp = EPSILON
            print(f"Note for FPR_p calculation: {g} group FP is 0, replaced with epsilon {EPSILON}")
        if tn == 0:
            tn = EPSILON
            print(f"Note for FPR_p calculation: {g} group TN is 0, replaced with epsilon {EPSILON}")

        table_fpr.append([fp, tn])
        denom = cm['FP'] + cm['TN']
        results['fpr_by_group'][g] = cm['FP'] / denom if denom > 0 else 0.0

    if len(table_fpr) >= 2:
        try:
            _, p_fpr, _, _ = chi2_contingency(table_fpr, correction=False)
            results['fpr_chi2_p'] = p_fpr
        except Exception as e:
            print(f"FPR chi-square test failed: {str(e)}")
            results['fpr_chi2_p'] = None

    # ------------------- 1.3. FNR Chi-square Test -------------------
    table_fnr = []
    for g in groups:
        cm = confusion_matrices_dict[g]
        fn = cm['FN']
        tp = cm['TP']

        # If FN or TP is 0, replace with epsilon
        if fn == 0:
            fn = EPSILON
            print(f"Note for FNR_p calculation: {g} group FN is 0, replaced with epsilon {EPSILON}")
        if tp == 0:
            tp = EPSILON
            print(f"Note for FNR_p calculation: {g} group TP is 0, replaced with epsilon {EPSILON}")

        table_fnr.append([fn, tp])

        # Calculate and store FNR for each group (using original values)
        denom = cm['FN'] + cm['TP']
        results['fnr_by_group'][g] = cm['FN'] / denom if denom > 0 else 0.0

    if len(table_fnr) >= 2:
        try:
            _, p_fnr, _, _ = chi2_contingency(table_fnr, correction=False)
            results['fnr_chi2_p'] = p_fnr
        except Exception as e:
            print(f"FNR chi-square test failed: {str(e)}")
            results['fnr_chi2_p'] = None

    # ------------------- 1.4. F1-score Chi-square Test -------------------
    table_f1 = []
    for g in groups:
        cm = confusion_matrices_dict[g]
        tp = cm['TP']
        errors = cm['FP'] + cm['FN']  # Incorrectly handled positive cases

        # If TP or errors is 0, replace with epsilon
        if tp == 0:
            tp = EPSILON
            print(f"Note for F1_p calculation: {g} group TP is 0, replaced with epsilon {EPSILON}")
        if errors == 0:
            errors = EPSILON
            print(f"Note for F1_p calculation: {g} group error count is 0, replaced with epsilon {EPSILON}")

        table_f1.append([tp, errors])

        # Calculate and store F1-score for each group (using original values)
        precision = cm['TP'] / (cm['TP'] + cm['FP']) if (cm['TP'] + cm['FP']) > 0 else 0.0
        recall = cm['TP'] / (cm['TP'] + cm['FN']) if (cm['TP'] + cm['FN']) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        results['f1_by_group'][g] = f1

    if len(table_f1) >= 2:
        try:
            _, p_f1, _, _ = chi2_contingency(table_f1, correction=False)
            results['f1_chi2_p'] = p_f1
        except Exception as e:
            print(f"F1 chi-square test failed: {str(e)}")
            results['f1_chi2_p'] = None

    # ------------------- 2. AOD (Average TPR and FPR Difference) -------------------
    tprs, fprs = [], []
    for g in groups:
        cm = confusion_matrices_dict[g]
        TP, FP, TN, FN = cm['TP'], cm['FP'], cm['TN'], cm['FN']
        denom_pos = TP + FN
        denom_neg = FP + TN
        TPR = TP / denom_pos if denom_pos > 0 else 0.0
        FPR = FP / denom_neg if denom_neg > 0 else 0.0
        tprs.append(TPR)
        fprs.append(FPR)
    if n >= 2:
        avg_tpr_diff = sum(abs(tprs[i] - tprs[j]) for i in range(n) for j in range(i + 1, n)) * 2 / (n * (n - 1))
        avg_fpr_diff = sum(abs(fprs[i] - fprs[j]) for i in range(n) for j in range(i + 1, n)) * 2 / (n * (n - 1))
        results['AOD'] = 0.5 * (avg_tpr_diff + avg_fpr_diff)

    # =================== 3. DI/EOP/EOD Calculations ===================
    # Prepare data needed for inter-group metric calculations
    group_tpr = {}  # Store TPR for each group (P(Ŷ=1|Y=1,A=i))
    group_fpr = {}  # Store FPR for each group (P(Ŷ=1|Y=0,A=i))
    for g in groups:
        cm = confusion_matrices_dict[g]
        TP, FP, TN, FN = cm['TP'], cm['FP'], cm['TN'], cm['FN']
        # Calculate TPR (P(Ŷ=1|Y=1)) - used for DI and EOP
        denom_pos = TP + FN
        tpr = TP / denom_pos if denom_pos > 0 else 0.0
        group_tpr[g] = tpr
        results['DI'][g] = tpr  # DI needs P(Ŷ=1|Y=1,A=i)
        # Calculate FPR (P(Ŷ=1|Y=0)) - used for EOD
        denom_neg = FP + TN
        fpr = FP / denom_neg if denom_neg > 0 else 0.0
        group_fpr[g] = fpr
        # Save to results
        results['EOP'][g] = tpr
        results['EOD'][g] = {'TPR': tpr, 'FPR': fpr}

    # ------------------- DI Calculation -------------------
    if n >= 2:
        di_vals = []  # Store all pairwise DI_ij values
        for i in range(n):
            for j in range(i + 1, n):
                P_i = group_tpr[groups[i]]
                P_j = group_tpr[groups[j]]

                # Avoid division by zero
                if P_i == 0 and P_j == 0:
                    ratio = 1.0  # Consider equal when both groups are 0
                elif P_i == 0 or P_j == 0:
                    ratio = 0.0  # Ratio is 0 when only one group is 0
                else:
                    ratio = min(P_i / P_j, P_j / P_i)
                di_vals.append(ratio)

        # Take the maximum of all pairwise DI_ij as global DI (Formula S2)
        results['overall_DI'] = max(di_vals) if di_vals else 1.0
    else:
        results['overall_DI'] = 1.0  # Perfect DI when only one group

    # ------------------- EOP Calculation -------------------
    if n >= 2:
        eop_sum = 0
        pair_count = 0
        for i in range(n):
            for j in range(i + 1, n):
                diff = abs(group_tpr[groups[i]] - group_tpr[groups[j]])
                eop_sum += diff
                pair_count += 1

        # Calculate average (Formula S3)
        results['overall_EOP'] = (2 / (n * (n - 1))) * eop_sum
    else:
        results['overall_EOP'] = 0.0  # No difference when only one group

    # ------------------- EOD Calculation -------------------
    if n >= 2:
        eod_sum = 0
        for i in range(n):
            for j in range(i + 1, n):
                # Calculate differences under y=0 and y=1 conditions (Formula S4)
                diff_y1 = abs(group_tpr[groups[i]] - group_tpr[groups[j]])
                diff_y0 = abs(group_fpr[groups[i]] - group_fpr[groups[j]])
                eod_sum += diff_y1 + diff_y0

        # Calculate average
        results['overall_EOD'] = (2 / (n * (n - 1))) * eod_sum
    else:
        results['overall_EOD'] = 0.0  # No difference when only one group

    return results


def analyze_fairness_by_groups(patient_ids, patient_true, patient_pred, patient_prob,
                               age_map, gender_map, analysis_name=""):
    """
    Unified fairness analysis function to avoid code duplication

    Args:
        patient_ids: List of patient IDs
        patient_true: True labels
        patient_pred: Predicted labels
        patient_prob: Predicted probabilities
        age_map: Age mapping dictionary
        gender_map: Gender mapping dictionary
        analysis_name: Analysis name (for printing)
    Returns:
        dict: Results containing metrics by group and fairness metrics
    """
    results = {
        'age_auc_by_group': {},
        'age_acc_by_group': {},
        'age_f1_by_group': {},
        'age_confusion_matrices': {},
        'gender_auc_by_group': {},
        'gender_acc_by_group': {},
        'gender_f1_by_group': {},
        'gender_confusion_matrices': {},
        'age_fairness': {},
        'gender_fairness': {}
    }

    print(f"\n=== {analysis_name} Fairness Analysis ===")

    # Ensure input data is in correct format
    patient_true = np.array(patient_true).flatten()
    patient_pred = np.array(patient_pred).flatten()
    patient_prob = np.array(patient_prob).flatten()

    # Age group analysis
    if age_map:
        # Dynamically get age groups, not hard-coded
        age_groups_present = set(age_map[pid] for pid in patient_ids if pid in age_map and age_map[pid] != -1)
        print(f"\nAge Group Analysis - Found age groups: {sorted(age_groups_present)}")
        for age_group in age_groups_present:
            # Get patient indices for this age group
            group_indices = [i for i, pid in enumerate(patient_ids) if age_map.get(pid) == age_group]
            if len(group_indices) < 2:  # Sample size check
                print(f"  Age group {age_group} has insufficient samples ({len(group_indices)}), skipping analysis")
                continue

            # Extract data for this group
            group_true = [patient_true[i] for i in group_indices]
            group_pred = [patient_pred[i] for i in group_indices]
            group_prob = [patient_prob[i] for i in group_indices]

            # Calculate metrics
            if len(set(group_true)) > 1:  # Ensure two classes exist to calculate AUC
                results['age_auc_by_group'][f'AgeGroup_{age_group}'] = roc_auc_score(group_true, group_prob)
            else:
                results['age_auc_by_group'][f'AgeGroup_{age_group}'] = np.nan
            results['age_acc_by_group'][f'AgeGroup_{age_group}'] = accuracy_score(group_true, group_pred)
            results['age_f1_by_group'][f'AgeGroup_{age_group}'] = f1_score(group_true, group_pred, zero_division=0)

            # Confusion matrix
            cm = confusion_matrix(group_true, group_pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                results['age_confusion_matrices'][f'AgeGroup_{age_group}'] = {
                    'TP': int(tp), 'FP': int(fp), 'TN': int(tn), 'FN': int(fn),
                    'samples': len(group_true)
                }
            # ================================Added on evening of 8/2=======================================
            elif cm.shape == (1, 1):
                # Case with only one class
                if group_true[0] == 0:  # All negative class
                    results['age_confusion_matrices'][f'AgeGroup_{age_group}'] = {
                        'TP': 0, 'FP': 0, 'TN': int(cm[0, 0]), 'FN': 0,
                        'samples': len(group_true)
                    }
                else:  # All positive class
                    results['age_confusion_matrices'][f'AgeGroup_{age_group}'] = {
                        'TP': int(cm[0, 0]), 'FP': 0, 'TN': 0, 'FN': 0,
                        'samples': len(group_true)
                    }
            # ===============================================================================

    # Gender group analysis (similar logic)
    if gender_map:
        gender_groups_present = set(
            gender_map[pid] for pid in patient_ids if pid in gender_map and gender_map[pid] != -1)
        print(f"\nGender Group Analysis - Found gender groups: {sorted(gender_groups_present)}")
        for gender_group in gender_groups_present:
            group_indices = [i for i, pid in enumerate(patient_ids) if gender_map.get(pid) == gender_group]

            if len(group_indices) < 2:
                print(f"  Gender group {gender_group} has insufficient samples ({len(group_indices)}), skipping analysis")
                continue

            group_true = [patient_true[i] for i in group_indices]
            group_pred = [patient_pred[i] for i in group_indices]
            group_prob = [patient_prob[i] for i in group_indices]

            if len(set(group_true)) > 1:
                results['gender_auc_by_group'][f'Gender_{gender_group}'] = roc_auc_score(group_true, group_prob)
            else:
                results['gender_auc_by_group'][f'Gender_{gender_group}'] = np.nan
            results['gender_acc_by_group'][f'Gender_{gender_group}'] = accuracy_score(group_true, group_pred)
            results['gender_f1_by_group'][f'Gender_{gender_group}'] = f1_score(group_true, group_pred, zero_division=0)

            cm = confusion_matrix(group_true, group_pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                results['gender_confusion_matrices'][f'Gender_{gender_group}'] = {
                    'TP': int(tp), 'FP': int(fp), 'TN': int(tn), 'FN': int(fn),
                    'samples': len(group_true)
                }
            # ================================Added on evening of 8/2=======================================
            elif cm.shape == (1, 1):
                # Case with only one class
                if group_true[0] == 0:  # All negative class
                    results['gender_confusion_matrices'][f'Gender_{gender_group}'] = {
                        'TP': 0, 'FP': 0, 'TN': int(cm[0, 0]), 'FN': 0,
                        'samples': len(group_true)
                    }
                else:  # All positive class
                    results['gender_confusion_matrices'][f'Gender_{gender_group}'] = {
                        'TP': int(cm[0, 0]), 'FP': 0, 'TN': 0, 'FN': 0,
                        'samples': len(group_true)
                    }
            # ===============================================================================

    # Calculate fairness metrics
    if results['age_confusion_matrices']:
        age_confusion_for_fairness = {k: {k2: v2 for k2, v2 in v.items() if k2 != 'samples'}
                                      for k, v in results['age_confusion_matrices'].items()}
        results['age_fairness'] = compute_fairness_metrics(age_confusion_for_fairness)

    if results['gender_confusion_matrices']:
        gender_confusion_for_fairness = {k: {k2: v2 for k2, v2 in v.items() if k2 != 'samples'}
                                         for k, v in results['gender_confusion_matrices'].items()}
        results['gender_fairness'] = compute_fairness_metrics(gender_confusion_for_fairness)

    # Print results
    print_fairness_results(results, analysis_name)
    return results


def print_fairness_results(results, analysis_name):
    """Unified function to print fairness analysis results"""

    # Print age group results
    if results['age_acc_by_group']:
        print(f"\n{analysis_name} - Age Group Accuracy:")
        for group, acc in results['age_acc_by_group'].items():
            auc = results['age_auc_by_group'].get(group, 'N/A')
            auc_str = f"{auc:.4f}" if not np.isnan(auc) else "N/A"
            f1 = results['age_f1_by_group'].get(group, 0.0)

            cm_info = results['age_confusion_matrices'].get(group, {})
            if cm_info:
                tp, fp, tn, fn = cm_info['TP'], cm_info['FP'], cm_info['TN'], cm_info['FN']
                samples = cm_info['samples']
                print(f"  {group}: ACC={acc:.4f}, AUC={auc_str}, F1={f1:.4f}, Samples={samples}")
                print(f"    Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
            else:
                print(f"  {group}: ACC={acc:.4f}, AUC={auc_str}, F1={f1:.4f}")

    # Print age group fairness metrics
    if results['age_fairness']:
        print(f"\n{analysis_name} - Age Group Fairness Metrics:")
        age_fairness = results['age_fairness']
        print(f"  Global Accuracy p-value: {age_fairness['accuracy_chi2_p']:.4f}")
        print(f"  FPR p-value: {age_fairness['fpr_chi2_p']:.4f}")
        print(f"  FNR p-value: {age_fairness['fnr_chi2_p']:.4f}")
        print(f"  F1 score p-value: {age_fairness['f1_chi2_p']:.4f}")
        print(f"  AOD: {age_fairness['AOD']:.4f}")
        print(f"  DI: {age_fairness['overall_DI']:.4f}")
        print(f"  EOP: {age_fairness['overall_EOP']:.4f}")
        print(f"  EOD: {age_fairness['overall_EOD']:.4f}")

    # Print gender group results
    if results['gender_acc_by_group']:
        print(f"\n{analysis_name} - Gender Group Accuracy:")
        for group, acc in results['gender_acc_by_group'].items():
            auc = results['gender_auc_by_group'].get(group, 'N/A')
            auc_str = f"{auc:.4f}" if not np.isnan(auc) else "N/A"
            f1 = results['gender_f1_by_group'].get(group, 0.0)

            cm_info = results['gender_confusion_matrices'].get(group, {})
            if cm_info:
                tp, fp, tn, fn = cm_info['TP'], cm_info['FP'], cm_info['TN'], cm_info['FN']
                samples = cm_info['samples']
                print(f"  {group}: ACC={acc:.4f}, AUC={auc_str}, F1={f1:.4f}, Samples={samples}")
                print(f"    Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
            else:
                print(f"  {group}: ACC={acc:.4f}, AUC={auc_str}, F1={f1:.4f}")

    # Print gender group fairness metrics
    if results['age_fairness']:
        print(f"\n{analysis_name} - Gender Group Fairness Metrics:")
        gender_fairness = results['gender_fairness']
        print(f"  Global Accuracy p-value: {gender_fairness['accuracy_chi2_p']:.4f}")
        print(f"  FPR p-value: {gender_fairness['fpr_chi2_p']:.4f}")
        print(f"  FNR p-value: {gender_fairness['fnr_chi2_p']:.4f}")
        print(f"  F1 score p-value: {gender_fairness['f1_chi2_p']:.4f}")
        print(f"  AOD: {gender_fairness['AOD']:.4f}")
        print(f"  DI: {gender_fairness['overall_DI']:.4f}")
        print(f"  EOP: {gender_fairness['overall_EOP']:.4f}")
        print(f"  EOD: {gender_fairness['overall_EOD']:.4f}")

# In[11]:
def load_patient_metadata(dataset='chinese'):
    """
    Load patient metadata (age and gender information)
    Args:
        dataset: 'chinese' or 'german'
    Returns:
        age_group_map: {patient_id: age_group} dictionary
        gender_map: {patient_id: gender_code} dictionary
    """
    age_group_map = {}
    gender_map = {}

    if dataset == 'chinese':
        meta_df = pd.read_csv('D:/data/audio/EENT/Patient_Metadata.csv', encoding='utf-8-sig')
    elif dataset == 'german':
        meta_df = pd.read_csv('D:/data/audio/SVD/German_patient_metadata.csv', encoding='utf-8-sig')
    else:
        raise ValueError("dataset must be 'chinese' or 'german'")

    # Process metadata
    for _, row in meta_df.iterrows():
        pid = row['ID']
        age = row['Age']
        gender = row['Gender']

        # Age grouping: <35 years=0, 35-50 years=1, >50 years=2
        if age < 35:
            age_group = 0
        elif 35 <= age <= 50:
            age_group = 1
        else:
            age_group = 2
        age_group_map[pid] = age_group

        # Gender encoding: Male=0, Female=1
        gender_code = 0 if gender.lower() == 'm' else 1
        gender_map[pid] = gender_code

    return age_group_map, gender_map


def build_sensitive_attrs_dict(patient_ids, age_map, gender_map):
    """
    Build sensitive attributes dictionary

    Args:
        patient_ids: List of patient IDs
        age_map: Age mapping dictionary
        gender_map: Gender mapping dictionary

    Returns:
        sensitive_attrs: {patient_id: {'age_group': int, 'gender': int}} dictionary
    """
    sensitive_attrs = {}
    for pid in patient_ids:
        sensitive_attrs[pid] = {
            'age_group': age_map.get(pid, -1),
            'gender': gender_map.get(pid, -1)
        }
    return sensitive_attrs

# In[12]:
def generate_model_name(model_type, vowel_type, dev_test_seed, train_val_seed, model_init_seed):
    """
    Generate standardized model name
    Args:
        model_type: 'cnn' or 'gmm'
        vowel_type: 'a' or 'i'
        dev_test_seed: Development/test set split seed
        train_val_seed: Training/validation set split seed
        model_init_seed: Model initialization seed
    Returns:
        model_name: Standardized model name
    Example:
        generate_model_name('cnn', 'a', 0, 300, 314)
        Returns: 'cnn_a_dev0_train300_init314'
    """
    return f"{model_type}_{vowel_type}_dev{dev_test_seed}_train{train_val_seed}_init{model_init_seed}"

def save_model_simple(model, model_name, model_type='auto', save_dir='./saved_models'):
    """
    Simple model saving function
    Args:
        model: Model to save
        model_name: Model name (without extension)
        model_type: Model type ('cnn', 'gmm', 'auto')
        save_dir: Save directory
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    # Automatically detect model type
    if model_type == 'auto':
        if hasattr(model, 'save'):  # Keras/TensorFlow model
            model_type = 'cnn'
        elif hasattr(model, 'predict_proba'):  # sklearn model
            model_type = 'gmm'
        else:
            raise ValueError("Unable to automatically detect model type, please specify manually")
    # Save model
    if model_type == 'cnn':
        model_path = os.path.join(save_dir, f"{model_name}.h5")
        model.save(model_path)
    elif model_type == 'gmm':
        model_path = os.path.join(save_dir, f"{model_name}.pkl")
        joblib.dump(model, model_path)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    print(f"Model saved: {model_path}")

def load_model_simple(model_name, model_type='auto', save_dir='./saved_models'):
    """
    Simple model loading function
    Args:
        model_name: Model name (without extension)
        model_type: Model type ('cnn', 'gmm', 'auto')
        save_dir: Model directory
    Returns:
        loaded_model: Loaded model
    """
    # Automatically detect model type
    if model_type == 'auto':
        cnn_path = os.path.join(save_dir, f"{model_name}.h5")
        gmm_path = os.path.join(save_dir, f"{model_name}.pkl")
        if os.path.exists(cnn_path):
            model_type = 'cnn'
        elif os.path.exists(gmm_path):
            model_type = 'gmm'
        else:
            raise FileNotFoundError(f"Model file not found: {model_name}")
    # Load model
    if model_type == 'cnn':
        import tensorflow as tf
        model_path = os.path.join(save_dir, f"{model_name}.h5")
        model = tf.keras.models.load_model(model_path)
    elif model_type == 'gmm':
        model_path = os.path.join(save_dir, f"{model_name}.pkl")
        model = joblib.load(model_path)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    print(f"Model loaded: {model_path}")
    return model

def save_model_config_simple(config_dict, config_name, save_dir='./saved_models'):
    """
    Save model configuration information
    Args:
        config_dict: Configuration dictionary
        config_name: Configuration file name
        save_dir: Save directory
    """
    os.makedirs(save_dir, exist_ok=True)
    config_path = os.path.join(save_dir, f"{config_name}_config.pkl")
    with open(config_path, 'wb') as f:
        pickle.dump(config_dict, f)
    print(f"Configuration saved: {config_path}")

def load_model_config_simple(config_name, save_dir='./saved_models'):
    """
    Load model configuration information
    Args:
        config_name: Configuration file name
        save_dir: Configuration directory
    Returns:
        config_dict: Configuration dictionary
    """
    config_path = os.path.join(save_dir, f"{config_name}_config.pkl")
    with open(config_path, 'rb') as f:
        config_dict = pickle.load(f)
    print(f"Configuration loaded: {config_path}")
    return config_dict

def list_saved_models(save_dir='./saved_models'):
    """
    List saved models
    Args:
        save_dir: Model directory
    Returns:
        models_info: Model information dictionary
    """
    if not os.path.exists(save_dir):
        print(f"Model directory does not exist: {save_dir}")
        return {}
    models_info = {}
    # Find all model files
    for file in os.listdir(save_dir):
        if file.endswith('.h5'):
            model_name = file[:-3]  # Remove .h5
            models_info[model_name] = {'type': 'cnn', 'file': file}
        elif file.endswith('.pkl') and not file.endswith('_config.pkl'):
            model_name = file[:-4]  # Remove .pkl
            models_info[model_name] = {'type': 'gmm', 'file': file}
    return models_info

# In[13]:
def model_eval_by_id(x_test, y_test, id_test, model, n_classes=2,
                     ROC=1, batch_size=None, strategy='best_threshold',
                     segment_threshold=0.5, percentage=0.2, vowel_type=None,
                     method=None, positive_group=None,
                     sensitive_attrs=None, dataset_type="Test"):
    """
    Evaluate model performance at patient level with various aggregation strategies

    Args:
        x_test: Test features
        y_test: Test labels
        id_test: Patient IDs for test samples
        model: Trained model to evaluate
        n_classes: Number of classes (default 2)
        ROC: Class index for ROC calculation (default 1)
        batch_size: Batch size for prediction (not used in current implementation)
        strategy: Aggregation strategy for patient-level prediction
        segment_threshold: Threshold for segment-level predictions
        percentage: Minimum percentage of positive segments for 'percentage' strategy
        vowel_type: Vowel type for reporting
        method: Model type ('gmm', 'CNN', or auto-detected)
        positive_group: Positive group index for GMM models
        sensitive_attrs: Sensitive attributes for fairness analysis
        dataset_type: Dataset type for reporting (e.g., "Test", "Validation")

    Returns:
        Dictionary containing evaluation results including metrics, ROC, and fairness analysis
    """

    # Data preprocessing
    if y_test.shape[1] == 1:
        new_y_test = np.zeros((len(y_test), 2))
        for i in range(len(new_y_test)):
            new_y_test[i, 0] = 1 - y_test[i, 0]
            new_y_test[i, 1] = y_test[i, 0]
        y_test = new_y_test

    y_test_lbl = np.argmax(y_test, axis=1)

    # Auto-detect model type
    if method is None:
        if hasattr(model, 'n_components') and hasattr(model, 'predict_proba'):
            method = 'gmm'
        elif hasattr(model, 'predict') and hasattr(model, 'layers'):
            method = 'CNN'
        else:
            method = 'unknown'

    # Model prediction
    if method == 'gmm':
        # GMM model processing
        y_pred_raw = model.predict_proba(x_test)
        # Rearrange according to positive_group to ensure column 1 is positive class probability
        y_pred = np.zeros_like(y_pred_raw)

        if positive_group == 1:
            y_pred[:, 0] = y_pred_raw[:, 0]  # Negative class probability
            y_pred[:, 1] = y_pred_raw[:, 1]  # Positive class probability
        else:  # positive_group == 0
            y_pred[:, 0] = y_pred_raw[:, 1]  # Negative class probability
            y_pred[:, 1] = y_pred_raw[:, 0]  # Positive class probability
    else:
        # CNN or other model processing
        if hasattr(model, 'predict_proba'):
            y_pred = model.predict_proba(x_test)
        else:
            y_pred = model.predict(x_test)

        # Ensure prediction format is correct
        if y_pred.ndim == 1:
            new_y_pred = np.zeros((len(y_pred), 2))
            for i in range(len(new_y_pred)):
                new_y_pred[i, 0] = 1 - y_pred[i]
                new_y_pred[i, 1] = y_pred[i]
            y_pred = new_y_pred
        elif y_pred.shape[1] == 1:
            new_y_pred = np.zeros((len(y_pred), 2))
            for i in range(len(new_y_pred)):
                new_y_pred[i, 0] = 1 - y_pred[i, 0]
                new_y_pred[i, 1] = y_pred[i, 0]
            y_pred = new_y_pred

    # Build patient data structure in one pass
    patient_data = {}  # {patient_id: {'segments': [indices], 'true_label': int, ...}}

    for i, patient_id in enumerate(id_test):
        if patient_id not in patient_data:
            patient_data[patient_id] = {
                'segments': [],
                'true_label': y_test_lbl[i],
                'segment_probs': [],
                'segment_preds': [],
                'pos_segments': 0,
                'neg_segments': 0
            }

        # Calculate segment-level prediction
        segment_prob = y_pred[i, 1]  # Positive class probability
        segment_pred = 1 if segment_prob >= segment_threshold else 0

        # Store information
        patient_data[patient_id]['segments'].append(i)
        patient_data[patient_id]['segment_probs'].append(segment_prob)
        patient_data[patient_id]['segment_preds'].append(segment_pred)

        # Count positive and negative segments
        if segment_pred == 1:
            patient_data[patient_id]['pos_segments'] += 1
        else:
            patient_data[patient_id]['neg_segments'] += 1

    # ===== Calculate patient-level statistics =====
    patient_ids = list(patient_data.keys())
    for patient_id in patient_ids:
        data = patient_data[patient_id]
        # Calculate average probabilities
        data['avg_prob_0'] = np.mean([y_pred[i, 0] for i in data['segments']])
        data['avg_prob_1'] = np.mean([y_pred[i, 1] for i in data['segments']])
        data['avg_prob'] = data['avg_prob_1']

    # ===== Extract arrays for sklearn =====
    pt_test_lbl = np.array([patient_data[pid]['true_label'] for pid in patient_ids])
    pt_pred = np.array([[patient_data[pid]['avg_prob_0'], patient_data[pid]['avg_prob_1']]
                        for pid in patient_ids])

    # ===== ROC calculation and optimal threshold =====
    fpr, tpr, thresholds = roc_curve(pt_test_lbl, pt_pred[:, ROC], pos_label=ROC)
    roc_auc = auc(fpr, tpr)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    # ===== Calculate final predictions based on strategy =====
    pt_pred_lbl = []
    for patient_id in patient_ids:
        data = patient_data[patient_id]
        prob = data['avg_prob']
        pos_segments = data['pos_segments']
        neg_segments = data['neg_segments']
        total_segments = pos_segments + neg_segments

        if strategy == 'best_threshold':
            pred = 1 if prob >= optimal_threshold else 0
        elif strategy == 'relative':
            pred = 1 if pos_segments >= neg_segments else 0
        elif strategy == 'percentage':
            segment_percentage = pos_segments / total_segments if total_segments > 0 else 0
            pred = 1 if segment_percentage >= percentage else 0
        elif strategy == 'max recall':
            pred = 1 if pos_segments > 0 else 0
        elif strategy == 'guding':
            pred = 1 if prob >= 0.5 else 0

        pt_pred_lbl.append(pred)
        patient_data[patient_id]['final_pred'] = pred

    pt_pred_lbl = np.array(pt_pred_lbl)

    # ===== Calculate wrong predictions =====
    wrong_idx = [i for i, elem in enumerate(pt_pred_lbl) if elem != pt_test_lbl[i]]
    wrong_id = [patient_ids[i] for i in wrong_idx]

    # ===== Output results =====
    print(f'\nClassification report for single vowel using strategy: {strategy}')
    print(classification_report(pt_test_lbl, pt_pred_lbl))

    # Calculate and display complete performance metrics
    metrics = calculate_performance_metrics(pt_test_lbl, pt_pred_lbl)
    vowel_name = vowel_type if vowel_type else "unknown"
    title_prefix = f"{dataset_type} Single Vowel" if dataset_type != "Test" else "Single Vowel"
    print_performance_metrics(metrics, roc_auc, f"{title_prefix} /{vowel_name}/ Performance ({strategy} strategy)")

    # Plot confusion matrix heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.title(f'Confusion Matrix - Vowel /{vowel_name}/ ({strategy} strategy)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    # plt.show()

    if ROC is not None:
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic curve')
        plt.legend(loc="lower right")
        # plt.show()

    fairness_results = {}
    if sensitive_attrs is not None:
        # Build age and gender mappings
        age_map = {pid: sensitive_attrs[pid]['age_group'] for pid in sensitive_attrs}
        gender_map = {pid: sensitive_attrs[pid]['gender'] for pid in sensitive_attrs}

        # Call unified fairness analysis function
        fairness_results = analyze_fairness_by_groups(
            patient_ids=patient_ids,
            patient_true=pt_test_lbl,
            patient_pred=pt_pred_lbl,
            patient_prob=pt_pred[:, ROC],
            age_map=age_map,
            gender_map=gender_map,
            analysis_name=f"Single vowel /{vowel_name}/ evaluation"
        )

    return {
        'wrong_id': wrong_id,
        'roc_auc': roc_auc,
        'classification_report': classification_report(pt_test_lbl, pt_pred_lbl, output_dict=True),
        'confusion_matrix': metrics['confusion_matrix'].tolist(),
        'optimal_threshold': optimal_threshold,
        'metrics': metrics,
        **fairness_results
    }


# In[13]:
def get_roc_curve_auc_a_i(model_a, model_i, x_a, y_a, ids_a, x_i, y_i, ids_i, method,
                          a_positive_group=None, i_positive_group=None,
                          strategy='best_threshold', segment_threshold=0.5, percentage=0.2,
                          sensitive_attrs=None, dataset_type="Test"):
    """
    Calculate ROC curve and AUC value for combined vowels (a+i), supporting multiple decision strategies

    Args:
        model_a: Trained model for vowel 'a'
        model_i: Trained model for vowel 'i'
        x_a: Features for vowel 'a'
        y_a: Labels for vowel 'a'
        ids_a: Patient IDs for vowel 'a'
        x_i: Features for vowel 'i'
        y_i: Labels for vowel 'i'
        ids_i: Patient IDs for vowel 'i'
        method: Model type ('gmm' or other)
        a_positive_group: Positive group index for GMM model 'a'
        i_positive_group: Positive group index for GMM model 'i'
        strategy: Decision strategy for patient-level prediction
        segment_threshold: Threshold for segment-level predictions
        percentage: Minimum percentage for 'percentage' strategy
        sensitive_attrs: Sensitive attributes for fairness analysis
        dataset_type: Dataset type for reporting (e.g., "Test", "Validation")

    Returns:
        Dictionary containing evaluation results including AUC, ROC curve, and fairness metrics
    """
    if method == 'gmm':
        # Get raw GMM prediction results
        y_a_predict_raw = model_a.predict_proba(x_a)
        y_i_predict_raw = model_i.predict_proba(x_i)

        # Rearrange according to positive_group to ensure column 1 is positive class probability
        y_a_predict = np.zeros_like(y_a_predict_raw)
        y_i_predict = np.zeros_like(y_i_predict_raw)

        if a_positive_group == 1:
            y_a_predict[:, 0] = y_a_predict_raw[:, 0]  # Negative class probability
            y_a_predict[:, 1] = y_a_predict_raw[:, 1]  # Positive class probability
        else:  # a_positive_group == 0
            y_a_predict[:, 0] = y_a_predict_raw[:, 1]  # Negative class probability
            y_a_predict[:, 1] = y_a_predict_raw[:, 0]  # Positive class probability

        if i_positive_group == 1:
            y_i_predict[:, 0] = y_i_predict_raw[:, 0]  # Negative class probability
            y_i_predict[:, 1] = y_i_predict_raw[:, 1]  # Positive class probability
        else:  # i_positive_group == 0
            y_i_predict[:, 0] = y_i_predict_raw[:, 1]  # Negative class probability
            y_i_predict[:, 1] = y_i_predict_raw[:, 0]  # Positive class probability
    else:
        # CNN model uses predict
        y_a_predict = model_a.predict(x_a)
        y_i_predict = model_i.predict(x_i)

        # Ensure CNN prediction format is correct
        if y_a_predict.ndim == 2 and y_a_predict.shape[1] == 1:
            temp_a = np.zeros((len(y_a_predict), 2))
            temp_a[:, 0] = 1 - y_a_predict[:, 0]
            temp_a[:, 1] = y_a_predict[:, 0]
            y_a_predict = temp_a

        if y_i_predict.ndim == 2 and y_i_predict.shape[1] == 1:
            temp_i = np.zeros((len(y_i_predict), 2))
            temp_i[:, 0] = 1 - y_i_predict[:, 0]
            temp_i[:, 1] = y_i_predict[:, 0]
            y_i_predict = temp_i

    y_predict = np.concatenate((y_a_predict, y_i_predict))
    ids = np.concatenate((ids_a, ids_i))
    y = np.concatenate((y_a, y_i))

    # Collect all patient data in one pass
    patient_data = {}

    for i, pid in enumerate(ids):
        segment_prob = y_predict[i][-1]
        segment_pred = 1 if segment_prob >= segment_threshold else 0
        true_label = y[i][0]

        if pid not in patient_data:
            patient_data[pid] = {
                'probs': [],
                'true_label': true_label,
                'pos_segments': 0,
                'neg_segments': 0,
                'a_segments': [],
                'i_segments': []
            }

        patient_data[pid]['probs'].append(segment_prob)
        if segment_pred == 1:
            patient_data[pid]['pos_segments'] += 1
        else:
            patient_data[pid]['neg_segments'] += 1

    # Calculate average probability for each patient
    for pid in patient_data:
        patient_data[pid]['avg_prob'] = np.mean(patient_data[pid]['probs'])

    # Extract data for ROC calculation
    patient_ids = list(patient_data.keys())
    patient_prob = [patient_data[pid]['avg_prob'] for pid in patient_ids]
    patient_true = [patient_data[pid]['true_label'] for pid in patient_ids]

    # Calculate ROC curve and optimal threshold
    fpr, tpr, thresholds = roc_curve(patient_true, patient_prob, drop_intermediate=True)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    # Calculate predictions based on strategy
    patient_pred = []
    for pid in patient_ids:
        data = patient_data[pid]
        prob = data['avg_prob']
        pos_segments = data['pos_segments']
        neg_segments = data['neg_segments']
        total_segments = pos_segments + neg_segments

        if strategy == 'best_threshold':
            pred = 1 if prob >= optimal_threshold else 0
        elif strategy == 'relative':
            pred = 1 if pos_segments >= neg_segments else 0
        elif strategy == 'percentage':
            segment_percentage = pos_segments / total_segments if total_segments > 0 else 0
            pred = 1 if segment_percentage >= percentage else 0
        elif strategy == 'max recall':
            pred = 1 if pos_segments > 0 else 0
        elif strategy == 'guding':
            pred = 1 if prob >= 0.5 else 0
        else:  # argmax or default
            pred = 1 if prob >= 0.5 else 0

        patient_pred.append(pred)
        # Store prediction result in patient_data
        patient_data[pid]['final_pred'] = pred

    print(f'\nClassification report for combined vowels using strategy: {strategy}')
    if strategy == 'best_threshold':
        print(f'Optimal threshold: {optimal_threshold:.4f}')
    elif strategy == 'percentage':
        print(f'Percentage threshold: {percentage}')

    print(classification_report(patient_true, patient_pred))
    auc = roc_auc_score(patient_true, patient_prob)

    metrics = calculate_performance_metrics(patient_true, patient_pred)
    title_prefix = f"{dataset_type} Single Vowel" if dataset_type != "Test" else "Single Vowel"
    print_performance_metrics(metrics, auc, f"{title_prefix} Combined Vowel /a+i/ Performance ({strategy} strategy)")

    # Plot confusion matrix heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.title(f'Confusion Matrix - Combined Vowel /a+i/ ({strategy} strategy)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    # plt.show()

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - Combined Vowel /a+i/ ({strategy} strategy)')
    plt.legend(loc="lower right")
    # plt.show()
    print('AUC = {}'.format(auc))

    fairness_results = {}
    if sensitive_attrs is not None:
        # Build age and gender mappings
        age_map = {pid: sensitive_attrs[pid]['age_group'] for pid in sensitive_attrs}
        gender_map = {pid: sensitive_attrs[pid]['gender'] for pid in sensitive_attrs}

        # Call unified fairness analysis function
        fairness_results = analyze_fairness_by_groups(
            patient_ids=patient_ids,
            patient_true=patient_true,
            patient_pred=patient_pred,
            patient_prob=patient_prob,
            age_map=age_map,
            gender_map=gender_map,
            analysis_name=f"Combined vowels /a+i/ evaluation"
        )

    return {
        'auc': auc,
        'fpr': fpr,
        'tpr': tpr,
        'classification_report': classification_report(patient_true, patient_pred, output_dict=True),
        'confusion_matrix': metrics['confusion_matrix'].tolist(),
        'metrics': metrics,
        **fairness_results
    }


# In[13]:
def load_and_preprocess_data(pkl_read_folder, ftype='mel', test_size=0.2, verbose=True):
    """
    Unified data loading and preprocessing function that uses global random seeds
    to ensure all models use the same data splits.

    Special notes:
    - When DEV_TEST_SEED = 9999, use fixed development/test set split
    - Otherwise, use multi-dimensional stratified splitting based on seeds

    Args:
        pkl_read_folder: Directory containing pickle files
        ftype: Feature type (default 'mel')
        test_size: Validation set proportion (default 0.2)
        verbose: Whether to print detailed information (default True)

    Returns:
        Complete dataset tuple containing Chinese and German data
    """
    if verbose:
        print("=" * 60)
        print("Unified Data Loading and Preprocessing")
        print("=" * 60)
        print(f"Feature type: {ftype}")
        print(f"Validation set proportion: {test_size}")

        # Display split mode
        if DEV_TEST_SEED == FIXED_SPLIT_SEED:
            print(f"Split mode: Fixed split (DEV_TEST_SEED={DEV_TEST_SEED})")
        else:
            print(f"Split mode: Stratified random split (DEV_TEST_SEED={DEV_TEST_SEED})")

    # Load Chinese data
    if verbose:
        print("\nLoading Chinese data...")
    x_train_a, x_val_a, x_test_a, y_train_a, y_val_a, y_test_a, id_train_a, id_val_a, id_test_a, \
        x_train_i, x_val_i, x_test_i, y_train_i, y_val_i, y_test_i, id_train_i, id_val_i, id_test_i, \
        sensitive_attrs_train, sensitive_attrs_val, sensitive_attrs_test = get_data_pkl_both_a_i(pkl_read_folder, ftype,
                                                                                                 0.2, verbose=verbose)

    # Load German data
    if verbose:
        print("\nLoading German data...")
    x_ger_a, y_ger_a, id_ger_a, x_ger_i, y_ger_i, id_ger_i, sensitive_attrs_ger = get_data_pkl_both_a_i_ger(
        pkl_read_folder, ftype)

    if verbose:
        print(f"\nGerman data shapes:")
        print(f"  German set - Vowel a: {x_ger_a.shape}, Vowel i: {x_ger_i.shape}")
        print("\n" + "=" * 60)
        print("Data loading and preprocessing completed")
        print("=" * 60)

    return (x_train_a, x_val_a, x_test_a, y_train_a, y_val_a, y_test_a, id_train_a, id_val_a, id_test_a,
            x_train_i, x_val_i, x_test_i, y_train_i, y_val_i, y_test_i, id_train_i, id_val_i, id_test_i,
            x_ger_a, y_ger_a, id_ger_a, x_ger_i, y_ger_i, id_ger_i,
            sensitive_attrs_train, sensitive_attrs_val, sensitive_attrs_test, sensitive_attrs_ger)


# In[14]:
def apply_feature_selection_and_dropna(data, selected_features=None):
    """
    Apply feature selection and handle missing values (mainly for GMM pitch features)

    Args:
        data: Data tuple returned by load_and_preprocess_data
        selected_features: List of selected feature indices, no feature selection if None

    Returns:
        Processed data tuple
    """
    (x_train_a, x_val_a, x_test_a, y_train_a, y_val_a, y_test_a, id_train_a, id_val_a, id_test_a,
     x_train_i, x_val_i, x_test_i, y_train_i, y_val_i, y_test_i, id_train_i, id_val_i, id_test_i,
     x_ger_a, y_ger_a, id_ger_a, x_ger_i, y_ger_i, id_ger_i) = data

    # Apply feature selection
    if selected_features is not None:
        print(f"Applying feature selection: {selected_features}")
        x_train_a = x_train_a[:, selected_features]
        x_val_a = x_val_a[:, selected_features]
        x_test_a = x_test_a[:, selected_features]
        x_train_i = x_train_i[:, selected_features]
        x_val_i = x_val_i[:, selected_features]
        x_test_i = x_test_i[:, selected_features]
        x_ger_a = x_ger_a[:, selected_features]
        x_ger_i = x_ger_i[:, selected_features]

    # Handle missing values
    print("Processing missing values...")
    # Process missing values in Chinese dataset
    Row_is_not_null = dropna_in_pitches(x_train_a)
    x_train_a_dropna = x_train_a[Row_is_not_null]
    y_train_a_dropna = y_train_a[Row_is_not_null]
    id_train_a_dropna = np.array(id_train_a)[Row_is_not_null]

    Row_is_not_null = dropna_in_pitches(x_val_a)
    x_val_a_dropna = x_val_a[Row_is_not_null]
    y_val_a_dropna = y_val_a[Row_is_not_null]
    id_val_a_dropna = np.array(id_val_a)[Row_is_not_null]

    Row_is_not_null = dropna_in_pitches(x_test_a)
    x_test_a_dropna = x_test_a[Row_is_not_null]
    y_test_a_dropna = y_test_a[Row_is_not_null]
    id_test_a_dropna = np.array(id_test_a)[Row_is_not_null]

    Row_is_not_null = dropna_in_pitches(x_train_i)
    x_train_i_dropna = x_train_i[Row_is_not_null]
    y_train_i_dropna = y_train_i[Row_is_not_null]
    id_train_i_dropna = np.array(id_train_i)[Row_is_not_null]

    Row_is_not_null = dropna_in_pitches(x_val_i)
    x_val_i_dropna = x_val_i[Row_is_not_null]
    y_val_i_dropna = y_val_i[Row_is_not_null]
    id_val_i_dropna = np.array(id_val_i)[Row_is_not_null]

    Row_is_not_null = dropna_in_pitches(x_test_i)
    x_test_i_dropna = x_test_i[Row_is_not_null]
    y_test_i_dropna = y_test_i[Row_is_not_null]
    id_test_i_dropna = np.array(id_test_i)[Row_is_not_null]

    # Process missing values in German dataset
    Row_is_not_null = dropna_in_pitches(x_ger_a)
    x_ger_a_dropna = x_ger_a[Row_is_not_null]
    y_ger_a_dropna = y_ger_a[Row_is_not_null]
    id_ger_a_dropna = np.array(id_ger_a)[Row_is_not_null]

    Row_is_not_null = dropna_in_pitches(x_ger_i)
    x_ger_i_dropna = x_ger_i[Row_is_not_null]
    y_ger_i_dropna = y_ger_i[Row_is_not_null]
    id_ger_i_dropna = np.array(id_ger_i)[Row_is_not_null]

    print(f"Processed data shapes:")
    print(f"  Training set - Vowel a: {x_train_a_dropna.shape}, Vowel i: {x_train_i_dropna.shape}")
    print(f"  Validation set - Vowel a: {x_val_a_dropna.shape}, Vowel i: {x_val_i_dropna.shape}")
    print(f"  Chinese test set - Vowel a: {x_test_a_dropna.shape}, Vowel i: {x_test_i_dropna.shape}")
    print(f"  German test set - Vowel a: {x_ger_a_dropna.shape}, Vowel i: {x_ger_i_dropna.shape}")

    return (x_train_a_dropna, x_val_a_dropna, x_test_a_dropna, y_train_a_dropna, y_val_a_dropna, y_test_a_dropna,
            id_train_a_dropna, id_val_a_dropna, id_test_a_dropna,
            x_train_i_dropna, x_val_i_dropna, x_test_i_dropna, y_train_i_dropna, y_val_i_dropna, y_test_i_dropna,
            id_train_i_dropna, id_val_i_dropna, id_test_i_dropna,
            x_ger_a_dropna, y_ger_a_dropna, id_ger_a_dropna, x_ger_i_dropna, y_ger_i_dropna, id_ger_i_dropna)


