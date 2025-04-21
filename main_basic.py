import datetime
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.tools import get_data, downsampling


def run_app():
    from sklearn.datasets import load_breast_cancer
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    from tabpfn import TabPFNClassifier

    import torch

    if not torch.cuda.is_available():
        raise SystemError(
            'GPU device not found. For fast training, please enable GPU. See section above for instructions.')
    data_fp = os.environ["DATA_FP"]
    # data_fp = r"openml"
    X, y = get_data(data_type=data_fp)
    # Load data
    # X, y = load_breast_cancer(return_X_y=True)

    if len(y) > 10_000:
        df = pd.concat([X, y], axis=1)
        X_down, y_down = downsampling(df=df)
    else:
        X_down, y_down = X, y


    X_train, X_test, y_train, y_test = train_test_split(X_down, y_down, test_size=0.2, random_state=42)

    # Initialize a classifier
    clf = TabPFNClassifier()
    # clf = AutoTabPFNClassifier(max_time=30, device="cuda")
    print("Fitting")
    clf.fit(X_train, y_train)

    # Predict probabilities
    print(f"{datetime.datetime.now()}, Predicting")
    prediction_probabilities = clf.predict_proba(X_test)
    print(f"{datetime.datetime.now()}, ROC AUC:", roc_auc_score(y_test, prediction_probabilities[:, 1]))

    # Predict labels
    predictions = clf.predict(X_test)
    print(f"{datetime.datetime.now()}, Accuracy", accuracy_score(y_test, predictions))

    chunk_size = 10_000
    predictions = pd.DataFrame({})

    # Use tqdm for progress bar (optional)
    for start in tqdm(range(0, len(y), chunk_size)):
        end = start + chunk_size
        chunk = X.iloc[start:end]

        # Predict only on the features your model expects
        preds = clf.predict_proba(chunk)

        # Append predictions (or assign to a new column if needed)
        predictions = pd.concat([predictions, pd.DataFrame(preds)])

    # Combine all chunks back into a single Series or DataFrame
    print(f"{datetime.datetime.now()}, saving results")
    X = X.reset_index().join(predictions)
    X.to_parquet("./pfn_predictions.parquet")
    print("done")


if __name__ == "__main__":
    run_app()