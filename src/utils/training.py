"""Data preprocessing Pipelines and model training."""

from typing import Literal, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectPercentile,
    mutual_info_classif,
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from feature_engine.selection import DropCorrelatedFeatures


class DataPreprocessing:
    """Class for preprocessing of the features."""

    def __init__(
        self,
        continuous_descriptors: List[str],
        discrete_descriptors: List[str],
        fingerprint_features: List[str],
        pca_n_components: Optional[int] = None,
        var_threshold: Optional[
            Dict[Literal["continuous", "discrete", "fingerprint"], float]
        ] = None,
        corr_threshold: float = 0.8,
        select_percentile: Optional[
            Dict[Literal["discrete", "fingerprint"], int]
        ] = None,
    ):
        """Initialize preprocessing Pipelines."""
        self.continuous_descriptors = continuous_descriptors
        self.discrete_descriptors = discrete_descriptors
        self.fingerprint_features = fingerprint_features

        var_threshold = var_threshold if var_threshold is not None else {}
        select_percentile = select_percentile if select_percentile is not None else {}

        self.continuous_preprocessing = Pipeline(
            steps=[  # DropCorrelatedFeatures needs to be first since it takes a DataFrame as an input
                (
                    "drop_corr_features",
                    DropCorrelatedFeatures(
                        variables=continuous_descriptors,
                        threshold=corr_threshold,
                    ),
                ),
                (
                    "drop_zero_var",
                    VarianceThreshold(threshold=var_threshold.get("continuous", 0.0)),
                ),
                (
                    "normalize",
                    StandardScaler(),
                ),  # pca assumes mean=0 and variance=1
                ("pca", PCA(n_components=pca_n_components)),
            ]
        )

        self.discrete_preprocessing = Pipeline(
            steps=[
                (
                    "drop_zero_var",
                    VarianceThreshold(threshold=var_threshold.get("discrete", 0.0)),
                ),
                ("min_max_normalization", MinMaxScaler()),
                (
                    "select_percentile",
                    SelectPercentile(
                        score_func=mutual_info_classif,
                        percentile=select_percentile.get("discrete", 10),
                    ),
                ),
            ]
        )

        self.fingerprint_preprocessing = Pipeline(
            steps=[
                (
                    "drop_zero_var",
                    VarianceThreshold(threshold=var_threshold.get("fingerprint", 0.0)),
                ),
                (
                    "select_percentile",
                    SelectPercentile(
                        score_func=mutual_info_classif,
                        percentile=select_percentile.get("fingerprint", 10),
                    ),
                ),
            ]
        )

    def fit(self, x_data: pd.DataFrame, y_data: Union[np.array, pd.Series]) -> np.array:
        """Fit the preprocessors of `self` to `x_data` and `y_data`."""

        self.continuous_preprocessing = self.continuous_preprocessing.fit(
            x_data[self.continuous_descriptors], y_data
        )
        self.discrete_preprocessing = self.discrete_preprocessing.fit(
            x_data[self.discrete_descriptors], y_data
        )
        self.fingerprint_preprocessing = self.fingerprint_preprocessing.fit(
            x_data[self.fingerprint_features], y_data
        )

    def transform(self, x_data: pd.DataFrame) -> np.array:
        """Transforms `x_data` according to preprocessors of `self`."""

        x_continuous_prep = self.continuous_preprocessing.transform(
            x_data[self.continuous_descriptors]
        )
        x_discrete_prep = self.discrete_preprocessing.transform(
            x_data[self.discrete_descriptors]
        )
        x_fingerprint_prep = self.fingerprint_preprocessing.transform(
            x_data[self.fingerprint_features]
        )

        return np.concatenate(
            (x_continuous_prep, x_discrete_prep, x_fingerprint_prep), axis=1
        )


def plot_confusion_matrix(y_val, y_pred, model_name):
    """Plots confusion matrix and outputs different metrics."""
    print(f"Accuracy: {accuracy_score(y_val, y_pred)*100:.2f}%")
    ConfusionMatrixDisplay.from_predictions(y_val, y_pred, cmap="Blues")
    plt.title(f"{model_name}\nConfusion Matrix")
    plt.show()
