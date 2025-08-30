from common_utils import *
import os
import sys
import warnings

warnings.filterwarnings('ignore')
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.impute import SimpleImputer

import tensorflow as tf

tf.config.experimental.enable_op_determinism()  # Set TensorFlow determinism immediately
print(tf.test.is_built_with_cuda())  # Should return True
print(tf.config.list_physical_devices('GPU'))  # Should display available GPU list

# Check if GPU is available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Set GPU memory growth as needed
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        print(e)

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

import xgboost as xgb
from sklearn import svm
from sklearn.gaussian_process import GaussianProcessClassifier

import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Flatten, Conv2D, BatchNormalization, ReLU, MaxPooling1D, \
    MaxPooling2D, GlobalAveragePooling1D
from tensorflow.keras.utils import to_categorical

# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import normalize
import random

import librosa
import librosa.display
import xlwt
from collections import Counter

from sklearn.model_selection import StratifiedKFold  # Five-fold cross validation
from sklearn.model_selection import KFold
from scipy.stats import chi2_contingency
from collections import defaultdict  # Summary functions
from collections import defaultdict
import random

# Add fixed split seed constant
# FIXED_SPLIT_SEED = 9999  # Keep consistent with common_utils.py
print(f"=== Current Configuration Check ===")
print(f"DEV_TEST_SEED = {DEV_TEST_SEED}")
print(f"FIXED_SPLIT_SEED = {FIXED_SPLIT_SEED}")
print(f"Using fixed split: {DEV_TEST_SEED == FIXED_SPLIT_SEED}")


def load_test_data_only(pkl_read_folder, ftype, target_dev_test_seed=None):
    """
    Load only test data, skip train/validation split

    Args:
        pkl_read_folder: Path to pickle files directory
        ftype: Feature type ('mel' or 'pitch')
        target_dev_test_seed: Target seed for dev/test split, if None use current global setting

    Returns:
        Dictionary containing test data for Chinese and German datasets
    """
    print("Loading test data only (skipping train/validation split)...")

    # Save original seed settings
    original_dev_test_seed = globals().get('DEV_TEST_SEED', 0)
    original_train_val_seed = globals().get('TRAIN_VAL_SEED', 100)

    try:
        # If target seed is specified, set it temporarily
        if target_dev_test_seed is not None:
            globals()['DEV_TEST_SEED'] = target_dev_test_seed
            # Display split mode being used
            if target_dev_test_seed == FIXED_SPLIT_SEED:
                print(f" Using fixed patient split (DEV_TEST_SEED={FIXED_SPLIT_SEED})")
            else:
                print(f" Using random seed split (DEV_TEST_SEED={target_dev_test_seed})")
        else:
            print(f" Using current global settings (DEV_TEST_SEED={original_dev_test_seed})")

        # Load complete dataset
        all_data = load_and_preprocess_data(pkl_read_folder, ftype, test_size=0.2, verbose=False)

        # Extract only test-related data
        (_, _, x_test_a, _, _, y_test_a, _, _, id_test_a,
         _, _, x_test_i, _, _, y_test_i, _, _, id_test_i,
         x_ger_a, y_ger_a, id_ger_a, x_ger_i, y_ger_i, id_ger_i,
         _, _, sensitive_attrs_test, sensitive_attrs_ger) = all_data

        return {
            'chinese': {
                'x_test_a': x_test_a, 'y_test_a': y_test_a, 'id_test_a': id_test_a,
                'x_test_i': x_test_i, 'y_test_i': y_test_i, 'id_test_i': id_test_i,
                'sensitive_attrs': sensitive_attrs_test
            },
            'german': {
                'x_ger_a': x_ger_a, 'y_ger_a': y_ger_a, 'id_ger_a': id_ger_a,
                'x_ger_i': x_ger_i, 'y_ger_i': y_ger_i, 'id_ger_i': id_ger_i,
                'sensitive_attrs': sensitive_attrs_ger
            }
        }
    finally:
        # Restore original seed settings
        globals()['DEV_TEST_SEED'] = original_dev_test_seed
        globals()['TRAIN_VAL_SEED'] = original_train_val_seed


def get_training_data_for_gmm_if_needed(pkl_read_folder, ftype, selected_features, model_a, model_i, config_a, config_i,
                                        target_dev_test_seed=None):
    """
    Load training data only when needed to recalculate GMM positive class groups

    Args:
        pkl_read_folder: Path to pickle files directory
        ftype: Feature type ('mel' or 'pitch')
        selected_features: List of selected feature indices
        model_a: Trained GMM model for vowel a
        model_i: Trained GMM model for vowel i
        config_a: Configuration for vowel a model
        config_i: Configuration for vowel i model
        target_dev_test_seed: Target seed for consistent data split

    Returns:
        Tuple of (a_positive_group, i_positive_group) - the positive class groups for GMM models
    """
    # Check if positive class groups need to be recalculated
    need_recalculate = (
                config_a is None or 'positive_group' not in config_a or config_i is None or 'positive_group' not in config_i)
    if not need_recalculate:
        return config_a['positive_group'], config_i['positive_group']

    print(" Need to recalculate GMM positive class groups, temporarily loading training data...")

    # Save original seed settings
    original_dev_test_seed = globals().get('DEV_TEST_SEED', 0)
    original_train_val_seed = globals().get('TRAIN_VAL_SEED', 100)

    try:
        # If target seed is specified, set temporarily (ensure training data split matches training time)
        if target_dev_test_seed is not None:
            globals()['DEV_TEST_SEED'] = target_dev_test_seed

            if target_dev_test_seed == FIXED_SPLIT_SEED:
                print(f" Using fixed patient split to load training data")
            else:
                print(f" Using seed {target_dev_test_seed} to load training data")

        # Temporarily load training data
        all_data = load_and_preprocess_data(pkl_read_folder, ftype, test_size=0.2, verbose=False)

        # Extract training data
        (x_train_a, _, _, y_train_a, _, _, _, _, _,
         x_train_i, _, _, y_train_i, _, _, _, _, _,
         _, _, _, _, _, _,
         _, _, _, _) = all_data

        # Apply feature selection and null value handling
        data_only = (x_train_a, x_train_a, x_train_a, y_train_a, y_train_a, y_train_a,
                     [0], [0], [0],  # Placeholders
                     x_train_i, x_train_i, x_train_i, y_train_i, y_train_i, y_train_i,
                     [0], [0], [0],  # Placeholders
                     x_train_a, y_train_a, [0], x_train_i, y_train_i, [0])  # Placeholders
        processed_data = apply_feature_selection_and_dropna(data_only, selected_features)
        x_train_a_processed = processed_data[0]
        y_train_a_processed = processed_data[3]
        x_train_i_processed = processed_data[9]
        y_train_i_processed = processed_data[12]

        # Recalculate positive class groups
        def gmm_group_to_class_single(model, x, y):
            """Helper function to determine which GMM component corresponds to positive class"""
            y_predict = model.predict_proba(x)
            if ((y_predict[:, 0] - y) ** 2).sum() <= ((y_predict[:, 1] - y) ** 2).sum():
                return 0
            else:
                return 1

        a_positive_group = gmm_group_to_class_single(model_a, x_train_a_processed, y_train_a_processed)
        i_positive_group = gmm_group_to_class_single(model_i, x_train_i_processed, y_train_i_processed)
        print(f"✓ Recalculation completed: vowel a={a_positive_group}, vowel i={i_positive_group}")
        return a_positive_group, i_positive_group
    finally:
        # Restore original seed settings
        globals()['DEV_TEST_SEED'] = original_dev_test_seed
        globals()['TRAIN_VAL_SEED'] = original_train_val_seed


def parse_model_seeds_from_name(model_name):
    """
    Parse seed information from model name
    Args:
        model_name: Model name, e.g., "gmm_a_dev9999_train100_init314"
    Returns:
        dict: Dictionary containing various seeds, returns None if parsing fails
    """
    try:
        parts = model_name.split('_')
        if len(parts) >= 5:
            # Extract seed information
            dev_part = parts[2]  # "dev9999"
            train_part = parts[3]  # "train100"
            init_part = parts[4]  # "init314"

            # Parse numbers
            dev_test_seed = int(dev_part.replace('dev', ''))
            train_val_seed = int(train_part.replace('train', ''))
            model_init_seed = int(init_part.replace('init', ''))

            return {
                'dev_test_seed': dev_test_seed,
                'train_val_seed': train_val_seed,
                'model_init_seed': model_init_seed
            }
    except (ValueError, IndexError) as e:
        print(f" Unable to parse seed information from model name: {e}")

    return None


def evaluate_models_by_names_test_only(model_a_name, model_i_name, verbose_data_split=False):
    """
    Model evaluation version using only test data

    Args:
        model_a_name: Name of the vowel a model
        model_i_name: Name of the vowel i model
        verbose_data_split: Whether to output verbose data split information

    Returns:
        Dictionary containing evaluation results and metadata
    """
    print("=" * 80)
    print("Evaluate Specified Models - Test Data Only Version")
    print("=" * 80)
    print(f"Vowel a model: {model_a_name}")
    print(f"Vowel i model: {model_i_name}")

    # Detect model type from model name
    if model_a_name.startswith('cnn_'):
        model_type = 'cnn'
        ftype = 'mel'
    elif model_a_name.startswith('gmm_'):
        model_type = 'gmm'
        ftype = 'pitch'
    else:
        print("Error: Cannot identify model type from model name")
        return None

    print(f"✓ Detected model type: {model_type.upper()}")

    # Parse seed information from model name
    seeds_info = parse_model_seeds_from_name(model_a_name)
    target_dev_test_seed = None

    if seeds_info:
        target_dev_test_seed = seeds_info['dev_test_seed']
        print(f"✓ Parsed seed information from model name: {seeds_info}")

        if target_dev_test_seed == FIXED_SPLIT_SEED:
            print(f"Detected fixed split model (DEV_TEST_SEED={FIXED_SPLIT_SEED})")
        else:
            print(f"Detected random split model (DEV_TEST_SEED={target_dev_test_seed})")
    else:
        print(" Unable to parse seed information from model name, will use current global settings")

    # Load models
    try:
        model_a = load_model_simple(model_a_name)
        model_i = load_model_simple(model_i_name)
        print("Models loaded successfully")
    except Exception as e:
        print(f" Failed to load models: {e}")
        return None

    # Load configuration files
    def load_single_config(model_name):
        """Helper function to load configuration for a single model"""
        try:
            parts = model_name.split('_')
            if len(parts) >= 5:
                model_type = parts[0]
                vowel = parts[1]
                dev_part = parts[2]
                train_part = parts[3]
                init_part = parts[4]
                config_name = f"{model_type}_{vowel}_config_{dev_part}_{train_part}_{init_part}"
                config = load_model_config_simple(config_name)
                print(f"✓ Successfully loaded configuration: {config_name}")
                return config
            else:
                raise ValueError("Incorrect model name format")
        except Exception as e:
            print(f"⚠️ Unable to load configuration file: {e}")
            return None

    config_a = load_single_config(model_a_name)
    config_i = load_single_config(model_i_name)

    # Load only test data
    pkl_read_folder = 'D:/data/audio/final_pickle_files'
    test_data = load_test_data_only(pkl_read_folder, ftype, target_dev_test_seed)

    # Process test data based on model type
    if model_type == 'gmm':
        print("Processing GMM test data...")

        # Get feature selection configuration
        if config_a and 'selected_features' in config_a:
            selected_features = config_a['selected_features']
        else:
            selected_features = [2, 3, 9]
        print(f"✓ Feature selection: {selected_features}")

        # Apply feature selection and null value handling to test data
        chinese_data = test_data['chinese']
        german_data = test_data['german']

        # Process Chinese test data
        test_data_only = (
            chinese_data['x_test_a'], chinese_data['x_test_a'], chinese_data['x_test_a'],
            chinese_data['y_test_a'], chinese_data['y_test_a'], chinese_data['y_test_a'],
            chinese_data['id_test_a'], chinese_data['id_test_a'], chinese_data['id_test_a'],
            chinese_data['x_test_i'], chinese_data['x_test_i'], chinese_data['x_test_i'],
            chinese_data['y_test_i'], chinese_data['y_test_i'], chinese_data['y_test_i'],
            chinese_data['id_test_i'], chinese_data['id_test_i'], chinese_data['id_test_i'],
            german_data['x_ger_a'], german_data['y_ger_a'], german_data['id_ger_a'],
            german_data['x_ger_i'], german_data['y_ger_i'], german_data['id_ger_i']
        )

        processed_test_data = apply_feature_selection_and_dropna(test_data_only, selected_features)

        # Update test data with processed versions
        x_test_a = processed_test_data[0]
        y_test_a = processed_test_data[3]
        id_test_a = processed_test_data[6]
        x_test_i = processed_test_data[9]
        y_test_i = processed_test_data[12]
        id_test_i = processed_test_data[15]
        x_ger_a = processed_test_data[18]
        y_ger_a = processed_test_data[19]
        id_ger_a = processed_test_data[20]
        x_ger_i = processed_test_data[21]
        y_ger_i = processed_test_data[22]
        id_ger_i = processed_test_data[23]

        # Get positive class groups (load training data only when needed)
        a_positive_group, i_positive_group = get_training_data_for_gmm_if_needed(
            pkl_read_folder, ftype, selected_features, model_a, model_i, config_a, config_i, target_dev_test_seed)

    else:  # CNN
        print("Processing CNN test data...")
        chinese_data = test_data['chinese']
        german_data = test_data['german']

        # Extract data directly for CNN (no feature selection needed)
        x_test_a = chinese_data['x_test_a']
        y_test_a = chinese_data['y_test_a']
        id_test_a = chinese_data['id_test_a']
        x_test_i = chinese_data['x_test_i']
        y_test_i = chinese_data['y_test_i']
        id_test_i = chinese_data['id_test_i']
        x_ger_a = german_data['x_ger_a']
        y_ger_a = german_data['y_ger_a']
        id_ger_a = german_data['id_ger_a']
        x_ger_i = german_data['x_ger_i']
        y_ger_i = german_data['y_ger_i']
        id_ger_i = german_data['id_ger_i']

    # Get sensitive attributes for fairness analysis
    sensitive_attrs_test = test_data['chinese']['sensitive_attrs']
    sensitive_attrs_ger = test_data['german']['sensitive_attrs']

    # Validate data shapes
    print(f"\nData Validation:")
    print(f"  Chinese Test Set - Vowel a: {x_test_a.shape}, Vowel i: {x_test_i.shape}")
    print(f"  German Test Set - Vowel a: {x_ger_a.shape}, Vowel i: {x_ger_i.shape}")

    # Display data split information
    if target_dev_test_seed == FIXED_SPLIT_SEED:
        print(f"  Using test set from fixed patient split")
    elif target_dev_test_seed is not None:
        print(f"  Using test set from random split with seed {target_dev_test_seed}")

    # Evaluate Chinese test set
    print("\n" + "=" * 50)
    print("Evaluating Chinese Test Set")
    print("=" * 50)

    print("\nEvaluating Vowel a:")
    eval_kwargs_a = {
        'n_classes': 2, 'ROC': 1, 'batch_size': None,
        'strategy': 'best_threshold', 'segment_threshold': 0.5, 'percentage': 0.2,
        'vowel_type': 'a', 'method': model_type,
        'sensitive_attrs': sensitive_attrs_test, 'dataset_type': 'Test'
    }
    if model_type == 'gmm':
        eval_kwargs_a['positive_group'] = a_positive_group

    test_result_a = model_eval_by_id(x_test_a, y_test_a, list(id_test_a), model_a, **eval_kwargs_a)

    print("\nEvaluating Vowel i:")
    eval_kwargs_i = {
        'n_classes': 2, 'ROC': 1, 'batch_size': None,
        'strategy': 'best_threshold', 'segment_threshold': 0.5, 'percentage': 0.2,
        'vowel_type': 'i', 'method': model_type,
        'sensitive_attrs': sensitive_attrs_test, 'dataset_type': 'Test'
    }
    if model_type == 'gmm':
        eval_kwargs_i['positive_group'] = i_positive_group

    test_result_i = model_eval_by_id(x_test_i, y_test_i, list(id_test_i), model_i, **eval_kwargs_i)

    print("\nEvaluating Combined Vowels a+i:")
    combined_kwargs = {
        'method': model_type,
        'strategy': 'best_threshold', 'segment_threshold': 0.5, 'percentage': 0.2,
        'sensitive_attrs': sensitive_attrs_test, 'dataset_type': 'Test'
    }
    if model_type == 'gmm':
        combined_kwargs['a_positive_group'] = a_positive_group
        combined_kwargs['i_positive_group'] = i_positive_group

    combined_result_ch = get_roc_curve_auc_a_i(
        model_a, model_i, x_test_a, y_test_a, id_test_a,
        x_test_i, y_test_i, id_test_i, **combined_kwargs)
    auc_ch = combined_result_ch['auc']

    # Evaluate German test set
    print("\n" + "=" * 50)
    print("Evaluating German Test Set")
    print("=" * 50)

    print("\nEvaluating Vowel a:")
    eval_kwargs_a['sensitive_attrs'] = sensitive_attrs_ger
    ger_result_a = model_eval_by_id(x_ger_a, y_ger_a, list(id_ger_a), model_a, **eval_kwargs_a)

    print("\nEvaluating Vowel i:")
    eval_kwargs_i['sensitive_attrs'] = sensitive_attrs_ger
    ger_result_i = model_eval_by_id(x_ger_i, y_ger_i, list(id_ger_i), model_i, **eval_kwargs_i)

    print("\nEvaluating Combined Vowels a+i:")
    combined_kwargs['sensitive_attrs'] = sensitive_attrs_ger
    combined_result_ger = get_roc_curve_auc_a_i(
        model_a, model_i, x_ger_a, y_ger_a, id_ger_a,
        x_ger_i, y_ger_i, id_ger_i, **combined_kwargs)
    auc_ger = combined_result_ger['auc']

    # Output summary results
    print("\n" + "=" * 80)
    print(f"{model_type.upper()} Model Evaluation Results Summary")
    print("=" * 80)
    print(f"Vowel a model: {model_a_name}")
    print(f"Vowel i model: {model_i_name}")
    print("=" * 80)

    # Display data split information
    if target_dev_test_seed == FIXED_SPLIT_SEED:
        print(f"Data Split: Fixed split (DEV_TEST_SEED={FIXED_SPLIT_SEED})")
    elif target_dev_test_seed is not None:
        print(f"Data Split: Random split (DEV_TEST_SEED={target_dev_test_seed})")

    return {
        'model_type': model_type,
        'model_names': {'a': model_a_name, 'i': model_i_name},
        'data_split_type': 'fixed' if target_dev_test_seed == FIXED_SPLIT_SEED else 'random',
        'dev_test_seed': target_dev_test_seed
    }


def main():
    """Main function to run model evaluation with test data only"""
    print("Starting model evaluation (test data only version)...")

    '''
    # Evaluate models trained with fixed split
    print("\n Evaluating models trained with fixed split:")
    model_a_name = "cnn_a_dev9999_train300_init2718"  # Fixed split trained
    model_i_name = "cnn_i_dev9999_train100_init314"
    result = evaluate_models_by_names_test_only(model_a_name, model_i_name)
    '''

    # Evaluate models trained with random split for comparison
    print("\n Evaluating models trained with random split:")
    model_a_name_random = "cnn_a_dev8_train100_init2718"
    model_i_name_random = "cnn_i_dev8_train300_init2718"
    result_random = evaluate_models_by_names_test_only(model_a_name_random, model_i_name_random)

    print("Evaluation completed!")


if __name__ == "__main__":
    main()