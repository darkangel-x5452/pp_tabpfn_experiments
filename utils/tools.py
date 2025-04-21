import gc

import pandas as pd
import torch
from IPython.core.display import Markdown
from IPython.core.display_functions import display
from catboost import CatBoostClassifier
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from tabpfn import TabPFNClassifier
from xgboost import XGBClassifier

from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier


def get_data(data_type: str) -> tuple[pd.DataFrame, pd.Series]:
    if data_type == "openml":
        # Parkinson's Disease dataset: Predict Parkinson's disease presence
        # Features: Voice measurements (e.g., frequency, amplitude)
        # Samples: 195 cases
        df = fetch_openml('parkinsons')

        # Alternative datasets (commented for reference):

        # German Credit Fraud (ID: 31)
        # Samples: 1,000
        # Features: 20 (account info, credit history, employment)
        # Target: Good/Bad credit risk
        # df = fetch_openml(data_id=31)

        # Cholesterol dataset: Predict cholesterol levels
        # Features: Patient characteristics, medical measurements
        # Samples: 303 patients
        # Target: Cholesterol levels in mg/dl
        # df = fetch_openml('cholesterol', version=2, as_frame=True)

        # Primary Tumor dataset: Predict tumor type and size
        # Features: Patient symptoms, medical test results
        # Samples: 339 patients
        # Target: Tumor classification and size
        # df = fetch_openml('primary-tumor', version=1, as_frame=True) - too many classes!

        # Heart Disease dataset (Statlog): Predict presence of heart disease
        # Features: Clinical and test measurements
        # Samples: 270 patients
        # Target: Binary heart disease diagnosis
        # df = fetch_openml("heart-statlog", version=1)

        # Diabetes dataset: Predict diabetes presence
        # Features: Medical measurements, patient history
        # Samples: 768 patients
        # Target: Binary diabetes diagnosis
        # df = fetch_openml("diabetes", version=1)

        # Hypothyroid dataset: Predict thyroid condition
        # Features: Blood test results, patient symptoms
        # Samples: 3772 patients
        # Target: Thyroid condition classification
        # df = fetch_openml('hypothyroid')
        display(Markdown(df['DESCR']))
        X: pd.DataFrame = df.data
        y: pd.Series = df.target
        return X, y
    elif data_type.startswith(r"C:\_databases"):
        fp = data_type
        df = pd.read_parquet(fp)
        X = df.drop(columns=['label'])
        y = df["label"]
        return X, y
    else:
        raise ValueError("Not valid data type")


def downsampling(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    # Assume your label column is called 'label'
    # Split the data
    true_df = df[df['label'] == True]
    false_df = df[df['label'] == False]

    # Compute how many false rows to keep
    # target_size = len(df) // 1000  # 8,000,000
    target_size = 10000  # 8,000,000
    n_true = len(true_df)
    n_false_to_sample = target_size - n_true

    # Sample the false rows randomly
    false_sampled = false_df.sample(n=n_false_to_sample, random_state=42)
    downsampled_df = pd.concat([true_df, false_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)

    X = downsampled_df.drop(columns=['label'])
    y = downsampled_df["label"]

    return X, y


def get_cross_val_mean(
        model_name: str,
        cv: int,
        scoring: str,
        X: pd.DataFrame,
        y: pd.Series
) -> float:
    if model_name == 'TabPFN':
        model = TabPFNClassifier(random_state=42)
    elif model_name == 'RandomForest':
        model = RandomForestClassifier(random_state=42)
    elif model_name == 'XGBoost':
        static_params = {
            "objective": "binary:logistic",  # Binary classification
            "tree_method": "hist",  # Use 'hist' or 'gpu_hist' for faster training
            "device": "cuda",
            "verbosity": 1,
            "alpha": 0.5580846759653957,
            "lambda": 1.6889274444889288,
            "learning_rate": 0.6342486558587844,
            "max_bin": int(295.0),
            "max_depth": int(44.0),
            "min_child_weight": int(10.0),
            "scale_pos_weight": int(785.0),
        }
        model = XGBClassifier(random_state=42, **static_params)
    elif model_name == 'CatBoost':
        model = CatBoostClassifier(random_state=42, verbose=0, task_type='GPU', devices='0')
    elif model_name == 'AutoTabPFN':
        model = AutoTabPFNClassifier(max_time=30, device="cuda")
    else:
        raise ValueError("Invalid model type")
    mean = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=1, verbose=1).mean()

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return mean
