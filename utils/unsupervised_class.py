import os
import pickle
from typing import Union

# Setup Imports
import pandas as pd
import numpy as np

from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.inspection import DecisionBoundaryDisplay

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder
from IPython.display import display, Markdown, Latex

# Baseline Imports
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

import torch

from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier

class Unsupervised():
    def __init__(self):
        pass

    def basic(self):
        from tabpfn_extensions import unsupervised

        # Load and prepare breast cancer dataset
        df = load_breast_cancer(return_X_y=False)
        X, y = df['data'], df['target']
        feature_names = df['feature_names']

        # Initialize TabPFN models
        model_unsupervised = unsupervised.TabPFNUnsupervisedModel(
            tabpfn_clf=TabPFNClassifier(),
            tabpfn_reg=TabPFNRegressor()
        )

        # Select features for synthetic data generation
        # Example features: [mean texture, mean area, mean concavity]
        feature_indices = [4, 6, 12]

        # Run synthetic data generation experiment
        experiment = unsupervised.experiments.GenerateSyntheticDataExperiment(
            task_type='unsupervised'
        )

        results = experiment.run(
            tabpfn=model_unsupervised,
            X=torch.tensor(X),
            y=torch.tensor(y),
            attribute_names=feature_names,
            temp=1.0,  # Temperature parameter for sampling
            n_samples=X.shape[0] * 2,  # Generate twice as many samples as original data
            indices=feature_indices
        )