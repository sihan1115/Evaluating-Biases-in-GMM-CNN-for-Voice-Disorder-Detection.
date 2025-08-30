# This script trains a CNN model for the vowel 'i' classification task.
# It includes deterministic behavior settings, model building, training, and saving functionalities.

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

#RANDOM_SELECTION_SEED = 42  # Controls random selection seed

# ===== Core configuration: Manual mode selection =====
USE_RANDOM_SELECTION = True  # Change to False to switch to manual mode
# Manually specified seed combinations
MANUAL_TRAIN_VAL_SEED = 300  # Manually specified train/validation set seed
MANUAL_CNN_SEED = 2718       # Manually specified CNN seed

def select_seeds_for_vowel_i():
    """
    Select seed combinations for vowel 'i'
    Returns:
        tuple: (train_val_seed, cnn_seed)
    """
    TRAIN_VAL_SEEDS = [100, 300, 500]
    CNN_SEEDS = [314, 2718]

    if USE_RANDOM_SELECTION:
        # Random selection mode - skip vowel 'a' selection
        train_val_seed = random.choice(TRAIN_VAL_SEEDS)  # Vowel 'i' selection
        cnn_seed = random.choice(CNN_SEEDS)
        print(f"CNN vowel 'i' [Random Selection] seed combination:")
        print(f"  TRAIN_VAL_SEED: {train_val_seed}")
        print(f"  CNN_SEED: {cnn_seed}")

    else:
        # Manual specification mode
        train_val_seed = MANUAL_TRAIN_VAL_SEED
        cnn_seed = MANUAL_CNN_SEED
        # Validate manually specified seeds
        if train_val_seed not in TRAIN_VAL_SEEDS:
            raise ValueError(f"Manually specified TRAIN_VAL_SEED ({train_val_seed}) is not in valid range {TRAIN_VAL_SEEDS}")
        if cnn_seed not in CNN_SEEDS:
            raise ValueError(f"Manually specified CNN_SEED ({cnn_seed}) is not in valid range {CNN_SEEDS}")
        print(f"CNN vowel 'i' [Manual Specification] seed combination:")
        print(f"  TRAIN_VAL_SEED: {train_val_seed}")
        print(f"  CNN_SEED: {cnn_seed}")

    return train_val_seed, cnn_seed

# Select seeds
TRAIN_VAL_SEED_I, CNN_SEED_I = select_seeds_for_vowel_i()

# ===== Key modification: Environment variables must be set before importing TensorFlow =====
os.environ['PYTHONHASHSEED'] = str(CNN_SEED_I)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
print(f"Setting environment variable PYTHONHASHSEED = {CNN_SEED_I}")
# ===================================================

import tensorflow as tf
tf.config.experimental.enable_op_determinism()     # Immediately set TensorFlow determinism
print(tf.test.is_built_with_cuda())  # Should return True
print(tf.config.list_physical_devices('GPU')) # Should show list of available GPUs
# Check if GPU is available
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
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Flatten, Conv2D, BatchNormalization, ReLU, MaxPooling1D, MaxPooling2D, GlobalAveragePooling1D
from tensorflow.keras.utils import to_categorical

import random
import librosa
import librosa.display
import xlwt
from collections import Counter

from sklearn.model_selection import StratifiedKFold # Five-fold cross validation
from sklearn.model_selection import KFold
from scipy.stats import chi2_contingency
from collections import defaultdict     # Aggregation function
from collections import defaultdict
import random


def set_deterministic_behavior(seed=CNN_SEED_I):
    """
    Set completely deterministic behavior for reproducibility
    Args:
        seed (int): Random seed to use
    """
    print(f"Setting CNN original version deterministic behavior with seed: {seed}")
    random.seed(seed)         # 1. Set Python random seed
    np.random.seed(seed)      # 2. Set NumPy random seed
    tf.random.set_seed(seed)  # 3. Set TensorFlow random seed
    print(f"Random seed verification:")     # 4. Verify settings
    print(f"  Python random test: {[random.random() for _ in range(2)]}")
    # Reset to ensure subsequent consistency
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print("Deterministic settings completed")


def build_model_cnn2d_original(input_shape, n_classes=2,
                      k_size=3,
                      stride_size=2,
                      pool_size=2,
                      act_fn='relu',
                      dropout=0.5, lr=0.00001,
                      seed=CNN_SEED_I):
    """
    Build the original 2D CNN model for vowel classification
    Args:
        input_shape: Shape of input data
        n_classes (int): Number of classes
        k_size: Kernel size
        stride_size: Stride size
        pool_size: Pooling size
        act_fn (str): Activation function
        dropout (float): Dropout rate
        lr (float): Learning rate
        seed (int): Random seed
    Returns:
        model: Compiled Keras model
    """

    if n_classes == 2:
        out_act_fn = 'sigmoid'
    else:
        out_act_fn = 'softmax'

    tf.keras.backend.clear_session()
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    kernel_initializer = tf.keras.initializers.GlorotUniform(seed=seed)   # Create deterministic initializer
    bias_initializer = tf.keras.initializers.Zeros()

    with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
        model = keras.Sequential()

    model.add(Conv2D(filters=32,
                     kernel_size=k_size,
                     strides=stride_size,
                     padding='same',
                     activation=act_fn,
                     input_shape=input_shape,
                     kernel_initializer=kernel_initializer,
                     bias_initializer=bias_initializer
                     ))
    model.add(MaxPooling2D(pool_size=pool_size, padding='same'))

    model.add(Conv2D(filters=32,
                     kernel_size=k_size,
                     strides=stride_size,
                     padding='same',
                     activation=act_fn,
                     kernel_initializer=kernel_initializer,
                     bias_initializer=bias_initializer
                     ))
    model.add(MaxPooling2D(pool_size=pool_size, padding='same'))
    model.add(Dropout(dropout / 2, seed=seed))

    model.add(Conv2D(filters=64,
                     kernel_size=k_size,
                     strides=stride_size,
                     padding='same',
                     activation=act_fn,
                     kernel_initializer=kernel_initializer,
                     bias_initializer=bias_initializer
                     ))
    model.add(MaxPooling2D(pool_size=pool_size, padding='same'))

    model.add(Conv2D(filters=64,
                     kernel_size=k_size,
                     strides=stride_size,
                     padding='same',
                     activation=act_fn,
                     kernel_initializer=kernel_initializer,
                     bias_initializer=bias_initializer
                     ))
    model.add(MaxPooling2D(pool_size=pool_size, padding='same'))
    model.add(Flatten())

    model.add(Dense(128,kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,activation=act_fn))
    model.add(Dropout(dropout, seed=seed))

    if n_classes == 2:
        model.add(Dense(1,activation=out_act_fn,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer))
    else:
        model.add(Dense(n_classes,activation=out_act_fn,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer))

    optimizer = keras.optimizers.Adam(learning_rate=lr)
    loss = keras.losses.BinaryCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

def main():
    """CNN vowel 'i' model training"""
    print("=" * 60)
    print("CNN vowel 'i' model training - Random seed version")
    print("=" * 60)
    print(f"Random seed configuration: DEV_TEST={DEV_TEST_SEED}, TRAIN_VAL={TRAIN_VAL_SEED_I}, GMM_INIT={CNN_SEED_I}")
    if DEV_TEST_SEED == 9999:
        print("Using fixed test set patient division")
    else:
        print("Using seed-based random division")

    # Set deterministic behavior
    set_deterministic_behavior()
    # Setup GPU
    setup_gpu()
    # Data path configuration
    pkl_read_folder = 'D:/data/audio/final_pickle_files'
    n_classes = 2
    ftype = 'mel'
    print(f"Feature type: {ftype}")

    # Set global TRAIN_VAL_SEED
    original_train_val_seed = globals().get('TRAIN_VAL_SEED', 100)
    import common_utils
    common_utils.TRAIN_VAL_SEED = TRAIN_VAL_SEED_I
    print(f"Modified verification: {common_utils.TRAIN_VAL_SEED}")

    print("\nLoading Chinese and German data...")
    all_data = load_and_preprocess_data(pkl_read_folder, ftype, test_size=0.2)
    # Unpack all data results (including Chinese and German data and sensitive attributes)
    (x_train_a, x_val_a, x_test_a, y_train_a, y_val_a, y_test_a, id_train_a, id_val_a, id_test_a,
     x_train_i, x_val_i, x_test_i, y_train_i, y_val_i, y_test_i, id_train_i, id_val_i, id_test_i,
     x_ger_a, y_ger_a, id_ger_a, x_ger_i, y_ger_i, id_ger_i,
     sensitive_attrs_train, sensitive_attrs_val, sensitive_attrs_test, sensitive_attrs_ger) = all_data

    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Build and train CNN model for vowel 'i'
    print("\nBuilding and training CNN model for vowel 'i'...")
    set_deterministic_behavior(CNN_SEED_I)
    CNN_i = build_model_cnn2d_original(x_train_i[0].shape, n_classes=n_classes,
                                       k_size=(3, 3), stride_size=(1, 1), pool_size=(2, 3),
                                       act_fn='relu', dropout=0.5, lr=0.00001,
                                       seed=CNN_SEED_I)
    CNN_i.summary()

    with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
        history_i = CNN_i.fit(x_train_i, y_train_i,
                              epochs=1000, batch_size=64, verbose=2,
                              callbacks=[early_stop],
                              validation_data=(x_val_i, y_val_i),
                              shuffle=True,
                              workers=1,
                              use_multiprocessing=False)

    # Save model
    print("\nSaving vowel 'i' model...")
    model_name_i = generate_model_name('cnn', 'i', DEV_TEST_SEED, TRAIN_VAL_SEED_I, CNN_SEED_I)
    print(f"Model name: {model_name_i}")

    save_model_simple(CNN_i, model_name_i)

    # Save configuration information
    config_info = {
        'model_type': 'cnn',
        'vowel': 'i',
        'model_name': model_name_i,
        'seeds': {
            'dev_test_seed': DEV_TEST_SEED,
            'train_val_seed': TRAIN_VAL_SEED_I,
            'cnn_seed': CNN_SEED_I,
        },
        'feature_type': ftype,
        'architecture': {
            'k_size': (3, 3),
            'stride_size': (1, 1),
            'pool_size': (2, 3),
            'act_fn': 'relu',
            'dropout': 0.5,
            'lr': 0.00001
        },
        'training_epochs': len(history_i.history['loss'])
    }

    config_name = f"cnn_i_config_dev{DEV_TEST_SEED}_train{TRAIN_VAL_SEED_I}_init{CNN_SEED_I}"
    save_model_config_simple(config_info, config_name)

    print("CNN vowel 'i' model training completed!")
    print(f"Model file: {model_name_i}.h5")
    print(f"Configuration file: {config_name}_config.pkl")

    # Restore original TRAIN_VAL_SEED
    globals()['TRAIN_VAL_SEED'] = original_train_val_seed

    # Print deterministic verification information
    print("\nDeterministic settings verification:")
    print(f"PYTHONHASHSEED: {os.environ.get('PYTHONHASHSEED', 'Not Set')}")
    print(f"TF_DETERMINISTIC_OPS: {os.environ.get('TF_DETERMINISTIC_OPS', 'Not Set')}")
    print(f"TF_CUDNN_DETERMINISTIC: {os.environ.get('TF_CUDNN_DETERMINISTIC', 'Not Set')}")

    return {
        'model': CNN_i,
        'model_name': model_name_i,
        'seeds': {
            'train_val_seed': TRAIN_VAL_SEED_I,
            'cnn_seed': CNN_SEED_I
        }
    }

if __name__ == "__main__":
    main()
