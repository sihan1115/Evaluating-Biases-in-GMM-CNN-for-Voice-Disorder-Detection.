# This script trains a GMM (Gaussian Mixture Model) for the vowel 'a' classification task.
# It includes deterministic behavior settings, model training, and saving functionalities.

# In[1]:
from common_utils import *
from sklearn.mixture import GaussianMixture

#RANDOM_SELECTION_SEED = 42  # Controls random selection seed

# ===== Core configuration: Mode selection =====
# Set to True to use random selection, False to use manual specification
USE_RANDOM_SELECTION = True  # Change to False to switch to manual mode
MANUAL_TRAIN_VAL_SEED = 100   # Manually specified train/validation set seed
MANUAL_GMM_SEED = 2718        # Manually specified GMM seed

def select_seeds_for_vowel_a():
    """
    Select seed combinations for vowel 'a'
    Returns:
        tuple: (train_val_seed, gmm_seed)
    """
    TRAIN_VAL_SEEDS = [100, 300, 500]
    GMM_SEEDS = [314, 2718]
    if USE_RANDOM_SELECTION:
        # Random selection mode
        train_val_seed = random.choice(TRAIN_VAL_SEEDS)
        gmm_seed = random.choice(GMM_SEEDS)
        print(f"GMM vowel 'a' [Random Selection] seed combination:")
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

        print(f"GMM vowel 'a' [Manual Specification] seed combination:")
        print(f"  TRAIN_VAL_SEED: {train_val_seed}")
        print(f"  GMM_SEED: {gmm_seed}")

    return train_val_seed, gmm_seed

# Select seeds
TRAIN_VAL_SEED_A, GMM_SEED_A = select_seeds_for_vowel_a()

def set_all_random_seeds(seed):
    """
    Set all random seeds that may affect GMM training
    This function should be called before each GMM training
    Args:
        seed (int): Random seed to use
    """
    random.seed(seed)                           # 1. Python built-in random number generator
    np.random.seed(seed)                       # 2. NumPy random number generator
    os.environ['PYTHONHASHSEED'] = str(seed)   # 3. Set environment variable to ensure complete determinism
    print(f"All random seeds have been set to: {seed}")

# In[2]:
def gmm_group_to_class(model_a, x_a, y_a):
    """
    Determine which GMM group corresponds to the positive class
    Args:
        model_a: Trained GMM model
        x_a: Input features
        y_a: True labels
    Returns:
        int: Index of the positive group (0 or 1)
    """
    y_a_predict = model_a.predict_proba(x_a)  # dimension of predicted y is m X 2.

    # calculates sum of square residuals. The group with smaller SSR is the positive group.
    if ((y_a_predict[:, 0] - y_a) ** 2).sum() <= ((y_a_predict[:, 1] - y_a) ** 2).sum():
        a_positive_group = 0
    else:
        a_positive_group = 1

    # If a_positive_group = 0, that means the probabilities of group 0 from GMM prediction is for positive class.
    return a_positive_group

# In[3]:
def main():
    """GMM model main function"""
    print("=" * 60)
    print("GMM vowel 'a' model training - Random seed version")
    print("=" * 60)
    print(f"Random seed configuration: DEV_TEST={DEV_TEST_SEED}, TRAIN_VAL={TRAIN_VAL_SEED_A}, GMM_INIT={GMM_SEED_A}")
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
    set_all_random_seeds(GMM_SEED_A)

    # Set global TRAIN_VAL_SEED
    original_train_val_seed = globals().get('TRAIN_VAL_SEED', 100)

    import common_utils
    common_utils.TRAIN_VAL_SEED = TRAIN_VAL_SEED_A
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

    # Train GMM model for vowel 'a'
    set_all_random_seeds(GMM_SEED_A)
    print("Training GMM model for vowel 'a'...")
    gmm_a = GaussianMixture(n_components=2, random_state=GMM_SEED_A)
    gmm_a.fit(x_train_a)

    # Determine positive class group
    print("\nDetermining GMM positive class group...")
    a_positive_group = gmm_group_to_class(gmm_a, x_train_a, y_train_a)
    print(f"Vowel 'a' positive group: {a_positive_group}")

    # Save model
    print("\nSaving trained model...")
    model_name_a = generate_model_name('gmm', 'a', DEV_TEST_SEED, TRAIN_VAL_SEED_A, GMM_SEED_A)
    print(f"Model name: {model_name_a}")

    save_model_simple(gmm_a, model_name_a)

    # Save configuration information
    config_info = {
        'model_type': 'gmm',
        'vowel': 'a',
        'model_name': model_name_a,
        'seeds': {
            'dev_test_seed': DEV_TEST_SEED,
            'train_val_seed': TRAIN_VAL_SEED_A,
            'gmm_seed': GMM_SEED_A,
        },
        'feature_type': ftype,
        'selected_features': selected_features,
        'positive_group': a_positive_group,
        'model_params': {
            'n_components': 2,
            'random_state': GMM_SEED_A
        }
    }

    config_name = f"gmm_a_config_dev{DEV_TEST_SEED}_train{TRAIN_VAL_SEED_A}_init{GMM_SEED_A}"
    save_model_config_simple(config_info, config_name)

    print("GMM vowel 'a' model saving completed!")
    print(f"Model file: {model_name_a}.pkl")
    print(f"Configuration file: {config_name}_config.pkl")

    # Restore original TRAIN_VAL_SEED
    globals()['TRAIN_VAL_SEED'] = original_train_val_seed

    return {
        'model': gmm_a,
        'model_name': model_name_a,
        'positive_group': a_positive_group,
        'seeds': {
            'train_val_seed': TRAIN_VAL_SEED_A,
            'gmm_seed': GMM_SEED_A
        }
    }

if __name__ == "__main__":
    main()
