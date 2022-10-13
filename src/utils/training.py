"""Data preprocessing Pipelines and model training."""

from typing import Literal, Dict, Optional, Union, List
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectPercentile,
    mutual_info_classif,
    chi2,
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from feature_engine.selection import DropCorrelatedFeatures

from skopt import forest_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import dump
from skopt.callbacks import CheckpointSaver


class DataPreprocessing:
    """Class for preprocessing of the features."""

    def __init__(
        self,
        feature_groups,
        var_threshold: Optional[
            Dict[Literal["continuous", "discrete", "fingerprint"], float]
        ] = None,
        corr_threshold: float = 0.8,
        select_percentile: Optional[
            Dict[Literal["discrete", "fingerprint"], int]
        ] = None,
        score_func: Optional[Literal["chi2", "mutual_info_classif"]] = None,
    ):
        """Initialize preprocessing Pipelines."""
        self.feature_groups = feature_groups

        var_threshold = var_threshold if var_threshold is not None else {}
        select_percentile = select_percentile if select_percentile is not None else {}
        if score_func is None:
            score_func = mutual_info_classif
        elif score_func == "chi2":
            score_func = chi2
        else:
            score_func = mutual_info_classif

        self.continuous_preprocessing = Pipeline(
            steps=[  # DropCorrelatedFeatures needs to be first since it takes a DataFrame as an input
                (
                    "drop_corr_features",
                    DropCorrelatedFeatures(
                        variables=self.feature_groups.continuous,
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
                ("pca", PCA(n_components="mle")),
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
                        score_func=score_func,
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
                        score_func=score_func,
                        percentile=select_percentile.get("fingerprint", 10),
                    ),
                ),
            ]
        )

    def fit(self, x_data: pd.DataFrame, y_data: Union[np.array, pd.Series]) -> np.array:
        """Fit the preprocessors of `self` to `x_data` and `y_data`."""

        self.continuous_preprocessing = self.continuous_preprocessing.fit(
            x_data[self.feature_groups.continuous], y_data
        )
        self.discrete_preprocessing = self.discrete_preprocessing.fit(
            x_data[self.feature_groups.discrete], y_data
        )
        self.fingerprint_preprocessing = self.fingerprint_preprocessing.fit(
            x_data[self.feature_groups.fingerprint], y_data
        )

    def transform(self, x_data: pd.DataFrame) -> np.array:
        """Transforms `x_data` according to preprocessors of `self`."""

        x_continuous_prep = self.continuous_preprocessing.transform(
            x_data[self.feature_groups.continuous]
        )
        x_discrete_prep = self.discrete_preprocessing.transform(
            x_data[self.feature_groups.discrete]
        )
        x_fingerprint_prep = self.fingerprint_preprocessing.transform(
            x_data[self.feature_groups.fingerprint]
        )

        return np.concatenate(
            (x_continuous_prep, x_discrete_prep, x_fingerprint_prep), axis=1
        )


def plot_confusion_matrix(y_val, y_pred, model_name):
    """Plot confusion matrix and output different metrics."""
    print(f"Accuracy: {accuracy_score(y_val, y_pred)*100:.2f}%")
    ConfusionMatrixDisplay.from_predictions(y_val, y_pred, cmap="Blues")
    plt.title(f"{model_name}\nConfusion Matrix")
    plt.show()


class BayesianOptimization:
    """
    Run Bayesian Optimization on model + preprocessing hyperparameters by optimizing accuracy score.
    TODO use another metric?
    """

    def __init__(
        self,
        model,
        file_name: str,
        model_params: List[Union[Real, Integer]],
        datasets,
        feature_groups,
        checkpointed_results = None
    ):
        """Constructor for BayesianOptimization of the model `model`.

        :param model: Model to be optimized.
        :param model_params: Hyperparameters of `model` to be optimized.
        :param checkpointed_results: Results of previoud run to be continued.
        """

        self.x_train = datasets["train"].drop("Y", axis=1)
        self.y_train = datasets["train"]["Y"]
        self.x_val = datasets["val"].drop("Y", axis=1)
        self.y_val = datasets["val"]["Y"]
        self.model = model
        self.file_name = file_name
        self.model_params = model_params
        self.checkpointed_results = checkpointed_results if checkpointed_results is not None else {}

        self.feature_groups = feature_groups
        self.file_loc = f"bayesian_optimization/{self.file_name}"
        self.checkpoint_callback = CheckpointSaver(self.file_loc)

        self.dimensions = [
            *model_params,
            Real(name="var_threshold_continuous", low=0.0, high=0.05),
            Real(name="var_threshold_discrete", low=0.0, high=0.05),
            Real(name="var_threshold_fingerprint", low=0.0, high=0.05),
            Real(name="corr_threshold", low=0.7, high=0.95),
            Integer(name="select_percentile_discrete", low=5, high=50),
            Integer(name="select_percentile_fingerprint", low=5, high=50),
            Categorical(name="score_func", categories=["chi2", "mutual_info_classif"]),
        ]

    def optimize(self, n_calls=100, n_initial_points=10, acq_func="EI"):
        """
        Run the optimization.

        After each evaluation of the objective function a checkoint will be stored. 
        This allows training to be stopped and continued any time.
        """
        result = forest_minimize(
            self.objective,
            dimensions=self.dimensions,
            n_calls=n_calls,
            n_initial_points=n_initial_points,
            acq_func=acq_func,
            verbose=True,
            n_jobs=-1,
            base_estimator="RF",
            callback=[self.checkpoint_callback],
            x0=self.checkpointed_results.get("x_iters", None),
            y0=self.checkpointed_results.get("func_vals", None)
        )

        print("\nSave results at", self.file_loc)
        dump(result, self.file_loc)
        print("""Best parameters:""")
        for parameter, value in zip(self.dimensions, result.x):
            print(f"- {parameter.name} = {value}")

        return result

    def train_objective(self, params: list):
        """Train `self.model` + Preprocess data given `params`.

        :param params: List of parameters for the model and preprocessing pipeline.
        """
        (
            *model_params,
            var_threshold_continuous,
            var_threshold_discrete,
            var_threshold_fingerprint,
            corr_threshold,
            select_percentile_discrete,
            select_percentile_fingerprint,
            score_func
        ) = params

        # Preprocessing
        preprocessing_pipeline = DataPreprocessing(
            feature_groups=self.feature_groups,
            var_threshold={
                "continuous": var_threshold_continuous,
                "discrete": var_threshold_discrete,
                "fingerprint": var_threshold_fingerprint,
            },
            corr_threshold=corr_threshold,
            select_percentile={
                "discrete": select_percentile_discrete,
                "fingerprint": select_percentile_fingerprint,
            },
            score_func=score_func,
        )
        preprocessing_pipeline.fit(self.x_train, self.y_train)
        x_train_preprocessed = preprocessing_pipeline.transform(self.x_train)

        print(
            f"Number of features after preprocessing: \
            {x_train_preprocessed.shape[1]}/{self.x_train.shape[1]}"
        )

        # Model
        named_model_params = {}
        for idx, param in enumerate(model_params):
            named_model_params[self.model_params[idx].name] = param

        model = self.model(**named_model_params)
        model.fit(x_train_preprocessed, self.y_train)

        return preprocessing_pipeline, model

    def evaluate_model(self, preprocessing_pipeline, model):
        """Evaluate `self.model`."""
        x_val_preprocessed = preprocessing_pipeline.transform(self.x_val)
        y_pred = model.predict(x_val_preprocessed)

        return -1.0 * accuracy_score(self.y_val, y_pred)

    def optimization_func(self, params: list):
        """Fit and evaluate `self.model` and the preprocessing pipeline."""
        preprocessing_pipeline, model = self.train_objective(params)
        return self.evaluate_model(preprocessing_pipeline, model)

    def objective(self, params: list) -> float:
        """Fit and evaluate `self.model` and the preprocessing pipeline."""
        preprocessing_pipeline, model = self.train_objective(params)
        return self.evaluate_model(preprocessing_pipeline, model)

    def best_confusion_matrix(self, results):
        """Plot confusion matrix and output different metrics."""
        preprocessing_pipeline, model = self.train_objective(results.x)
        x_val_preprocessed = preprocessing_pipeline.transform(self.x_val)
        y_pred = model.predict(x_val_preprocessed)

        plot_confusion_matrix(self.y_val, y_pred, f"Optimized {self.file_name}")
