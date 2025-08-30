# file: /Users/sihanxie/Desktop/Effectiveness and Fairness/train_evaluate/gmm_i.py
# This script trains a GMM (Gaussian Mixture Model) for the vowel 'i' classification task.
# It includes deterministic behavior settings, model training, and saving functionalities.

# In[1]:
from common_utils import *
from sklearn.mixture import GaussianMixture

#RANDOM_SELECTION_SEED = 42  # Controls random selection seed

# ===== Core configuration: Manual mode selection =====
USE_RANDOM_SELECTION = False  # Change to False to switch to manual mode
# Manually specified seed combinations
MANUAL_TRAIN_VAL_SEED = 300   # Manually specified train/validation set seed
MANUAL_GMM_SEED = 2718        # Manually specified GMM seed

def select_seeds_for_vowel_i():
    """
    Select seed combinations for vowel 'i'
    Returns:
        tuple: (train_val_seed, gmm_seed)
    """
    TRAIN_VAL_SEEDS = [100, 300, 500]
    GMM_SEEDS = [314, 2718]

    if USE_RANDOM_SELECTION:
        # Random selection mode - skip vowel 'a' selection
        train_val_seed = random.choice(TRAIN_VAL_SEEDS)  # Vowel 'i' selection
        gmm_seed = random.choice(GMM_SEEDS)
        print(f"GMM vowel 'i' [Random Selection] seed combination:")
        print(f"  TRAIN_VAL_SEED: {train_val_seed}")
        print(f"  GMM_SEED: {gmm_seed}")

    else:
        # Manual specification mode
        train_val_seed = MANUAL_TRAIN_VAL_SEED
        gmm_seed = MANUAL_GMM_SEED
        # Validate manually specified seeds
        if train_val_seed not in TRAIN_VAL_SEEDS:
            raise ValueError(f"Manually specified TRAIN_VAL_SEED ({train_val_seed}) is not in valid range {TRAIN_VAL_SEEDS}")
        if gmm_seed not in GMM_SEEDS:
            raise ValueError(f"Manually specified GMM_SEED ({gmm_seed}) is not in valid range {GMM_SEEDS}")
        print(f"GMM vowel 'i' [Manual Specification] seed combination:")
        print(f"  TRAIN_VAL_SEED: {train_val_seed}")
        print(f"  GMM_SEED: {gmm_seed}")

    return train_val_seed, gmm_seed

# Select seeds
TRAIN_VAL_SEED_I, GMM_SEED_I = select_seeds_for_vowel_i()

# Set random seeds
def set_all_random_seeds(seed):
    """
    Set all random seeds that may affect GMM training
    Args:
        seed (int): Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"All random seeds have been set to: {seed}")

# In[2]:
def gmm_group_to_class(model_i, x_i, y_i):
    """
    Determine which GMM group corresponds to the positive class
    Args:
        model_i: Trained GMM model
        x_i: Input features
        y_i: True labels
    Returns:
        int: Index of the positive group (0 or 1)
    """
    y_i_predict = model_i.predict_proba(x_i)

    if ((y_i_predict[:, 0] - y_i) ** 2).sum() <= ((y_i_predict[:, 1] - y_i) ** 2).sum():
        i_positive_group = 0
    else:
        i_positive_group = 1

    return i_positive_group

# In[3]:
def main():
    """GMM vowel 'i' model training"""
    print("=" * 60)
    print("GMM vowel 'i' model training - Random seed version")
    print("=" * 60)
    print(f"Random seed configuration: DEV_TEST={DEV_TEST_SEED}, TRAIN_VAL={TRAIN_VAL_SEED_I}, GMM_INIT={GMM_SEED_I}")
    if DEV_TEST_SEED == 9999:
        print("Using fixed test set patient division")
    else:
        print("Using seed-based random division")

    # Data path configuration
    pkl_read_folder = 'D:/data/audio/final_pickle_files'
    n_classes = 2
    ftype = 'pitch'
    selected_features = [2, 3, 9]
    print(f"Feature type: {ftype}")

    # Set seeds
    set_all_random_seeds(GMM_SEED_I)

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

    print("\nPerforming feature selection and null value processing...")
    # Only perform feature selection and null value processing on data parts, preserving sensitive attributes
    data_only = (x_train_a, x_val_a, x_test_a, y_train_a, y_val_a, y_test_a, id_train_a, id_val_a, id_test_a,
                 x_train_i, x_val_i, x_test_i, y_train_i, y_val_i, y_test_i, id_train_i, id_val_i, id_test_i,
                 x_ger_a, y_ger_a, id_ger_a, x_ger_i, y_ger_i, id_ger_i)

    processed_data = apply_feature_selection_and_dropna(data_only, selected_features)

    # Unpack processed data
    (x_train_a, x_val_a, x_test_a, y_train_a, y_val_a, y_test_a, id_train_a, id_val_a, id_test_a,
     x_train_i, x_val_i, x_test_i, y_train_i, y_val_i, y_test_i, id_train_i, id_val_i, id_test_i,
     x_ger_a, y_ger_a, id_ger_a, x_ger_i, y_ger_i, id_ger_i) = processed_data

    print("\nTraining GMM vowel 'i' model...")
    # Train GMM model for vowel 'i'
    set_all_random_seeds(GMM_SEED_I)
    print("Training GMM model for vowel 'i'...")
    gmm_i = GaussianMixture(n_components=2, random_state=GMM_SEED_I)
    gmm_i.fit(x_train_i)

    # Determine positive class group
    print("\nDetermining GMM positive class group...")
    i_positive_group = gmm_group_to_class(gmm_i, x_train_i, y_train_i)
    print(f"Vowel 'i' positive group: {i_positive_group}")

    # Save model
    print("\nSaving trained model...")
    model_name_i = generate_model_name('gmm', 'i', DEV_TEST_SEED, TRAIN_VAL_SEED_I, GMM_SEED_I)
    print(f"Model name: {model_name_i}")

    save_model_simple(gmm_i, model_name_i)

    # Save configuration information
    config_info = {
        'model_type': 'gmm',
        'vowel': 'i',
        'model_name': model_name_i,
        'seeds': {
            'dev_test_seed': DEV_TEST_SEED,
            'train_val_seed': TRAIN_VAL_SEED_I,
            'gmm_seed': GMM_SEED_I,
        },
        'feature_type': ftype,
        'selected_features': selected_features,
        'positive_group': i_positive_group,
        'model_params': {
            'n_components': 2,
            'random_state': GMM_SEED_I
        }
    }

    config_name = f"gmm_i_config_dev{DEV_TEST_SEED}_train{TRAIN_VAL_SEED_I}_init{GMM_SEED_I}"
    save_model_config_simple(config_info, config_name)

    print("GMM vowel 'i' model saving completed!")
    print(f"Model file: {model_name_i}.pkl")
    print(f"Configuration file: {config_name}_config.pkl")

    # Restore original TRAIN_VAL_SEED
    globals()['TRAIN_VAL_SEED'] = original_train_val_seed

    return {
        'model': gmm_i,
        'model_name': model_name_i,
        'positive_group': i_positive_group,
        'seeds': {
            'train_val_seed': TRAIN_VAL_SEED_I,
            'gmm_seed': GMM_SEED_I
        }
    }

if __name__ == "__main__":
    main()
