# src/data.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


@dataclass
class IrisData:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    feature_names: List[str]
    target_names: List[str]


def load_iris_split(test_size: float = 0.2, random_state: int = 42) -> IrisData:
    iris = load_iris(as_frame=True)
    df = iris.frame  # includes features + 'target' column
    # The target column is always named 'target' when as_frame=True
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return IrisData(
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        feature_names=list(X.columns),
        target_names=list(iris.target_names),
    )
