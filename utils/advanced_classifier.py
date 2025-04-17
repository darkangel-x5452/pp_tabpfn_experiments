import datetime
import gc
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
import matplotlib.pyplot as plt

import torch

from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier


import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

from utils.tools import downsampling


class AdvancedClassifier():
    def __init__(self):
        if not torch.cuda.is_available():
            raise SystemError(
                'GPU device not found. For fast training, please enable GPU. See section above for instructions.')
        torch.cuda.empty_cache()
        gc.collect()

        self.models_path = "artifacts/models"
        self.model_fp = f'{self.models_path}/tabpfn_classifier.pkl'

    def setup_train_data(
            self,
            x: pd.DataFrame,
            y: pd.Series
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        X, y = x, y

        # Encode target labels to classes
        le = LabelEncoder()
        y = le.fit_transform(y)

        # Convert all categorical columns to numeric
        for col in X.select_dtypes(['category']).columns:
            X[col] = X[col].cat.codes

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        display(X)
        # if len(y) > 12_000:
        #     test_size = (1 - 10000 / len(y))
        # else:
        test_size = 0.20
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        return X_train, X_test, y_train, y_test

    def train_model(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
    ):
        model = TabPFNClassifier(random_state=42, ignore_pretraining_limits=True).fit(X_train, y_train)
        return model

    def evaluate_model(
            self,
            model: TabPFNClassifier,
            X_test: pd.DataFrame,
            y_test: pd.Series,
            num_class: int,
    ):
        with torch.no_grad():
            y_pred = model.predict_proba(X_test)
            # Calculate ROC AUC (handles both binary and multiclass)
            score = roc_auc_score(y_test, y_pred if num_class > 2 else y_pred[:, 1])
            print(f"TabPFN ROC AUC: {score:.4f}")

    def save_model(
            self,
            model: TabPFNClassifier,
    ):
        # Save the trained classifier to a file
        with open(self.model_fp, 'wb') as f:
            pickle.dump(model, f)

    def load_model(
            self,
    ) -> TabPFNClassifier:
        with open(self.model_fp, 'rb') as f:
            loaded_classifier = pickle.load(f)
        return loaded_classifier

    def compare_models(
            self,
            X: pd.DataFrame,
            y: pd.Series,

    ):
        # Compare different machine learning models by training each one multiple times
        # on different parts of the data and averaging their performance scores for a
        # more reliable performance estimate
        # ---- First set of models ----
        # Step 1: Model definitions
        scoring = 'roc_auc_ovr' if len(np.unique(y)) > 2 else 'roc_auc'
        if len(y) > 10_000:
            df = pd.concat([X, y], axis=1)
            X_down, y_down = downsampling(df=df)
        else:
            X_down, y_down = X, y
        # Encode target values to be 0, 1, 2, ...
        le = LabelEncoder()
        y_down = le.fit_transform(y_down)
        y = le.fit_transform(y)

        models1_down = [
            ('TabPFN', TabPFNClassifier(random_state=42)),
        ]
        print(f"{datetime.datetime.now()}, scoring 1 down")
        scores1 = {
            name: cross_val_score(model, X_down, y_down, cv=5, scoring=scoring, n_jobs=1, verbose=1).mean()
            for name, model in models1_down
        }
        models1 = [
            ('RandomForest', RandomForestClassifier(random_state=42)),
            ('XGBoost', XGBClassifier(random_state=42, device="cuda", tree_method="hist")),
            ('CatBoost', CatBoostClassifier(random_state=42, verbose=0, task_type='GPU', devices='0'))
        ]
        print(f"{datetime.datetime.now()},  1 normal")
        scores1.update({
            name: cross_val_score(model, X, y, cv=5, scoring=scoring, n_jobs=1, verbose=1).mean()
            for name, model in models1
        })
        df1 = pd.DataFrame(list(scores1.items()), columns=['Model', 'ROC AUC'])

        models2 = [
            ('TabPFN', TabPFNClassifier(random_state=42)),
            ('AutoTabPFN', AutoTabPFNClassifier(max_time=30, device="cuda")),
        ]
        print(f"{datetime.datetime.now()},  2 down")
        scores2 = {
            name: cross_val_score(model, X_down, y_down, cv=3, scoring=scoring, n_jobs=1, verbose=1).mean()
            for name, model in models2
        }
        models2 = [
            ('RandomForest', RandomForestClassifier(random_state=42)),
            ('XGBoost', XGBClassifier(random_state=42, device="cuda", tree_method="hist")),
            ('CatBoost', CatBoostClassifier(random_state=42, verbose=0, task_type='GPU', devices='0'))
        ]
        print(f"{datetime.datetime.now()},  2 normal")
        scores2.update({
            name: cross_val_score(model, X, y, cv=3, scoring=scoring, n_jobs=1, verbose=1).mean()
            for name, model in models2
        })
        print(f"{datetime.datetime.now()}, creating dataframe")
        df2 = pd.DataFrame(list(scores2.items()), columns=['Model', 'ROC AUC'])

        print(f"{datetime.datetime.now()}, plotting data")
        # Step 2: Create a side-by-side plot
        ylim_min = min(df1['ROC AUC'].min(), df2['ROC AUC'].min()) * 0.995
        ylim_max = min(1.0, max(df1['ROC AUC'].max(), df2['ROC AUC'].max()) * 1.005)

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Without AutoTabPFN (5-fold)", "With AutoTabPFN (3-fold)")
        )

        fig.add_trace(
            go.Bar(x=df1['Model'], y=df1['ROC AUC'], name='Without AutoTabPFN', marker_color='skyblue'),
            row=1, col=1
        )

        fig.add_trace(
            go.Bar(x=df2['Model'], y=df2['ROC AUC'], name='With AutoTabPFN', marker_color='salmon'),
            row=1, col=2
        )

        # Global layout
        fig.update_yaxes(range=[ylim_min, ylim_max], title='ROC AUC', row=1, col=1)
        fig.update_yaxes(range=[ylim_min, ylim_max], title='ROC AUC', row=1, col=2)
        fig.update_layout(title_text="Model Comparison")#, width=1000, height=500)

        # Show in browser
        pio.show(fig)

    def interpret_features(
            self,
    ):

        from tabpfn_extensions import interpretability

        # Load example dataset
        data = load_breast_cancer()
        X, y = data.data, data.target
        feature_names = data.feature_names
        n_samples = 50

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5
        )

        # Initialize and train model
        clf = TabPFNClassifier()
        clf.fit(X_train, y_train)

        # Calculate SHAP values
        shap_values = interpretability.shap.get_shap_values(
            estimator=clf,
            test_x=X_test[:n_samples],
            attribute_names=feature_names,
            algorithm="permutation",
        )

        # Create visualization
        fig = interpretability.shap.plot_shap(shap_values)

        from tabpfn_extensions import interpretability

        # Load data
        data = load_breast_cancer()
        X, y = data.data, data.target
        feature_names = data.feature_names

        # Initialize model
        clf = TabPFNClassifier(n_estimators=1)

        # Feature selection
        sfs = interpretability.feature_selection.feature_selection(
            estimator=clf,
            X=X,
            y=y,
            n_features_to_select=2,
            feature_names=feature_names
        )

        # Print selected features
        selected_features = [feature_names[i] for i in range(len(feature_names)) if sfs.get_support()[i]]
        print("\nSelected features:")
        for feature in selected_features:
            print(f"- {feature}")

    def playground_data_chart(
            self
    ):
        # Toy functions that generate the data
        def generate_circle(n_datapoints, radius, noise_factor):
            angles = np.linspace(0, 2 * np.pi, n_datapoints).T
            x = radius * np.cos(angles) + np.random.randn(n_datapoints) * noise_factor
            y = radius * np.sin(angles) + np.random.randn(n_datapoints) * noise_factor

            return np.stack([x, y]).T

        def generate_concentric_cirlces(radii, num_points_per_circle, noise_factor=1 / 15):
            circles = []
            for r, num_points in zip(radii, num_points_per_circle):
                circles.append(generate_circle(num_points, r, noise_factor))

            circle = np.vstack(circles)
            return circle

        def generate_circle_data(num_points_per_circle, radii, noise_factor):
            radii = np.array(radii)
            circles_1 = generate_concentric_cirlces(radii, num_points_per_circle, noise_factor)
            circles_1 = np.hstack([circles_1, np.zeros((sum(num_points_per_circle), 1))])

            circles_2 = generate_concentric_cirlces(radii + 0.3, num_points_per_circle, noise_factor)
            circles_2 = np.hstack([circles_2, np.ones((sum(num_points_per_circle), 1))])

            circles = np.vstack([circles_1, circles_2])
            X, y = circles[:, :2], circles[:, 2]
            return X, y

        # Generate the data
        X_train, y_train = generate_circle_data(
            num_points_per_circle=[50, 100, 200],
            radii=[1, 2, 4],
            noise_factor=0.1
        )

        # Function for plotting
        def plot_decision_boundary(ax, model, model_name):
            cmap = ListedColormap(["#FF0000", "#0000FF"])
            ax.set_title(model_name)
            DecisionBoundaryDisplay.from_estimator(
                model, X_train[:, :2], alpha=0.6, ax=ax, eps=0.2, grid_resolution=50, response_method="predict_proba",
                cmap=plt.cm.RdBu,
            )
            ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train > 0, cmap=cmap)

        rf = RandomForestClassifier().fit(X_train[:, :2], y_train)
        xgb = XGBClassifier().fit(X_train[:, :2], y_train)
        tabpfn = TabPFNClassifier().fit(X_train[:, :2], y_train)

        # Create a 2x2 subplot layout
        fig, axes = plt.subplots(2, 2, figsize=(7, 7))

        # Plot Train Points
        ax_points = axes[0, 0]
        ax_points.set_title("Train points")
        ax_points.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=ListedColormap(["#FF0000", "#0000FF"]))

        # Plot Random Forest
        ax_rf = axes[0, 1]
        plot_decision_boundary(ax_rf, rf, "Random Forest")

        # Plot XGBoost
        ax_xgb = axes[1, 0]
        plot_decision_boundary(ax_xgb, xgb, "XGBoost")

        # Plot TabPFN
        ax_tabpfn = axes[1, 1]
        plot_decision_boundary(ax_tabpfn, tabpfn, "TabPFN")

        # Adjust layout for better spacing
        plt.tight_layout()
        plt.show()