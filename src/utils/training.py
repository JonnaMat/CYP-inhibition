"""Data preprocessing Pipelines and model training."""

from typing import Literal, Dict, Optional, Union, List
from os.path import exists
import numpy as np
import pandas as pd
from feature_engine.selection import DropCorrelatedFeatures

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
from sklearn.dummy import DummyClassifier

# pylint: disable=wildcard-import
# pylint: disable=unused-wildcard-import
from sklearn.metrics import *

from skopt import forest_minimize
from skopt.space import Real, Integer, Categorical

PYLAB_PARAMS = {
    "legend.fontsize": 14,
    "figure.figsize": (7 * 1.6, 7),
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
}


class DataPreprocessing:
    """Class for preprocessing of the features."""

    def __init__(
        self,
        feature_groups,
        var_threshold: Optional[
            Dict[Literal["continuous", "discrete", "fingerprint"], float]
        ] = None,
        corr_threshold: float = 0.95,
        # select_percentile: Optional[
        #     Dict[Literal["discrete", "fingerprint"], int]
        # ] = None,
        # score_func: Optional[Literal["chi2", "mutual_info_classif"]] = None,
    ):
        """Initialize preprocessing Pipelines."""
        self.feature_groups = feature_groups

        var_threshold = var_threshold if var_threshold is not None else {}
        # select_percentile = select_percentile if select_percentile is not None else {}
        # if score_func is None:
        #     score_func = mutual_info_classif
        # elif score_func == "chi2":
        #     score_func = chi2
        # else:
        #     score_func = mutual_info_classif

        self.continuous_preprocessing = Pipeline(
            steps=[  # DropCorrelatedFeatures needs to be first since it takes a DataFrame as input
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
                    MinMaxScaler(),
                ),  # StandardScaler() if pca since pca assumes mean=0 and variance=1
                # ("pca", PCA(n_components="mle")),
            ]
        )

        self.discrete_preprocessing = Pipeline(
            steps=[
                (
                    "drop_zero_var",
                    VarianceThreshold(threshold=var_threshold.get("discrete", 0.0)),
                ),
                ("min_max_normalization", MinMaxScaler()),
                # (
                #     "select_percentile",
                #     SelectPercentile(
                #         score_func=score_func,
                #         percentile=select_percentile.get("discrete", 10),
                #     ),
                # ),
            ]
        )

        self.fingerprint_preprocessing = Pipeline(
            steps=[
                (
                    "drop_zero_var",
                    VarianceThreshold(threshold=var_threshold.get("fingerprint", 0.0)),
                ),
                # (
                #     "select_percentile",
                #     SelectPercentile(
                #         score_func=score_func,
                #         percentile=select_percentile.get("fingerprint", 10),
                #     ),
                # ),
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


def conf_matrix(y_val, y_pred, model_name):
    """Plot confusion matrix and output different metrics."""
    plt.style.use("default")
    pylab.rcParams.update(PYLAB_PARAMS)
    print(f"Accuracy: {accuracy_score(y_val, y_pred)*100:.2f}%")
    ConfusionMatrixDisplay.from_predictions(y_val, y_pred, cmap="Blues")
    plt.title(f"{model_name}\nConfusion Matrix")
    plt.show()


def plot_parameter_metric(
    model_name: str,
    metric: str,
    parameter: str,
    param_values: List[float],
    metric_values: List[float],
):
    """Plot metric vs parameter."""
    plt.style.use("seaborn-darkgrid")
    pylab.rcParams.update(PYLAB_PARAMS)
    plt.plot(param_values, metric_values)
    plt.title(f"{model_name}\n{metric} vs {parameter}")
    plt.xlabel(parameter)
    plt.xticks(ticks=param_values[::5])
    plt.ylabel(metric)
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
        model_params: List[Union[Real, Integer, Categorical]],
        datasets,
        feature_groups,
        preprocessing_params: Optional[List[Union[Real, Integer, Categorical]]] = None,
        fix_model_params: Optional[dict] = None,
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
        self.file_loc = f"optimization/{self.file_name}.csv"
        if exists(self.file_loc):
            self.results = pd.read_csv(self.file_loc, comment="#").drop(
                "Unnamed: 0", axis=1
            )
        else:
            self.file = open(self.file_loc, "w", encoding="utf-8")
            self.file.write(f"# {fix_model_params}\n")
            self.results = None
        self.run_counter = 0

        self.feature_groups = feature_groups

        self.model_params = model_params
        self.fix_model_params = fix_model_params if fix_model_params is not None else {}
        self.preprocessing_params = (
            preprocessing_params
            if preprocessing_params is not None
            else [
                Real(name="var_threshold_continuous", low=0.0, high=0.05),
                Real(name="var_threshold_discrete", low=0.0, high=0.05),
                Real(name="var_threshold_fingerprint", low=0.0, high=0.05),
                Real(name="corr_threshold", low=0.8, high=1.0),
                # Integer(name="select_percentile_discrete", low=5, high=50),
                # Integer(name="select_percentile_fingerprint", low=5, high=50),
                # Categorical(name="score_func", categories=["chi2", "mutual_info_classif"]),
            ]
        )
        self.dimensions = [*model_params, *self.preprocessing_params]
        self.metrics = [
            "accuracy",
            "f1",
            "precision",
            "recall",
            "mcc",
        ]
        predict_proba = getattr(self.model(), "predict_proba", None)
        if callable(predict_proba):
            self.metrics += ["roc_auc_score", "average_precision_score"]
            self.predict_proba = True
        else:
            self.predict_proba = False

    def optimize(self, n_calls=50, n_initial_points=10, acq_func="EI"):
        """
        Run the optimization.

        After each evaluation of the objective function a checkoint will be stored.
        This allows training to be stopped and continued any time.
        """
        if self.results is not None:
            return

        for dim in self.dimensions:
            self.file.write(f",{dim.name}")
        for metric in self.metrics:
            self.file.write(f",val_{metric}")
        self.file.write("\n")

        result = forest_minimize(
            self.objective,
            dimensions=self.dimensions,
            n_calls=n_calls,
            n_initial_points=n_initial_points,
            acq_func=acq_func,
            verbose=True,
            n_jobs=-1,
        )

        print("\nSave results at", self.file_loc)
        print("""Best parameters:""")
        for parameter, value in zip(self.dimensions, result.x):
            print(f"- {parameter.name} = {value}")

        self.file.close()
        self.results = pd.read_csv(self.file_loc, comment="#").drop(
            "Unnamed: 0", axis=1
        )

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
            # select_percentile_discrete,
            # select_percentile_fingerprint,
            # score_func
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
            # select_percentile={
            #     "discrete": select_percentile_discrete,
            #     "fingerprint": select_percentile_fingerprint,
            # },
            # score_func=score_func,
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

        model = self.model(**named_model_params, **self.fix_model_params)
        model.fit(x_train_preprocessed, self.y_train)

        return preprocessing_pipeline, model

    def evaluate_model(self, preprocessing_pipeline, model):
        """Evaluate `self.model`."""
        x_val_preprocessed = preprocessing_pipeline.transform(self.x_val)
        y_pred = model.predict(x_val_preprocessed)

        y_pred_proba = (
            model.predict_proba(x_val_preprocessed)[:, 1]
            if self.predict_proba
            else None
        )
        metrics = calculate_metrics(self.y_val, y_pred, y_pred_proba)

        for eval_metric_score in metrics.values():
            self.file.write(f",{eval_metric_score}")
        self.file.write("\n")

        return -1.0 * metrics["accuracy"]

    def optimization_func(self, params: list):
        """Fit and evaluate `self.model` and the preprocessing pipeline."""
        preprocessing_pipeline, model = self.train_objective(params)
        return self.evaluate_model(preprocessing_pipeline, model)

    def objective(self, params: list) -> float:
        """Fit and evaluate `self.model` and the preprocessing pipeline."""
        self.file.write(str(self.run_counter))
        self.run_counter += 1
        for param in params:
            self.file.write(f",{param}")

        preprocessing_pipeline, model = self.train_objective(params)
        score = self.evaluate_model(preprocessing_pipeline, model)

        return score

    def get_predictions(self, params: list):
        """Predictions of model given parameters."""
        preprocessing_pipeline, model = self.train_objective(params)
        x_val_preprocessed = preprocessing_pipeline.transform(self.x_val)
        y_pred = model.predict(x_val_preprocessed)
        y_pred_proba = (
            model.predict_proba(x_val_preprocessed)[:, 1]
            if self.predict_proba
            else None
        )
        return y_pred, y_pred_proba

    def pretty_results(self, **kwargs):
        """Return `self.results` with highlighted best results."""
        metric_columns = list(self.results.filter(regex="val_"))

        return pretty_print_df(
            self.results.sort_values("val_accuracy", ascending=False),
            subset=metric_columns,
            **kwargs
        )


def pretty_print_df(
    data_sorted: pd.DataFrame, subset: list, highlight_color="darkgreen", quantile=0.95
):
    """Return `data_sorted` with highlighted best results and quantile."""
    return data_sorted.head().style.highlight_quantile(
        subset=subset,
        props="font-weight:bold; color:#fffd75",
        axis=0,
        q_left=quantile,
    ).highlight_max(color=highlight_color, axis=0, subset=subset)


def get_baseline(datasets):
    """Plot the Confusion Matri of the Baseline."""
    baseline = DummyClassifier()
    baseline.fit(datasets["train"].drop("Y", axis=1), datasets["train"]["Y"])
    y_pred_baseline = baseline.predict(datasets["val"].drop("Y", axis=1))

    conf_matrix(datasets["val"]["Y"], y_pred_baseline, "Baseline - DummyClassifier")
    plt.show()


def calculate_metrics(y_true, y_pred, y_predict_proba: Optional[List] = None):
    """Return various metrics."""
    # metric_names = ["accuracy","f1","precision","recall","mcc"]
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    if y_predict_proba is not None:
        metrics["roc_auc_score"] = roc_auc_score(y_true, y_predict_proba)
        metrics["average_precision_score"] = average_precision_score(
            y_true, y_predict_proba
        )
    return metrics


def compare_metric_curves(classifiers: Dict[str, list], y_true):
    """
    Compare ROC; DET and Precision-recall curves of classifiers in `classifiers`.

    `classifiers` is of the form: `{str(name of classifier) : y_predict_proba}`.
    """
    # prepare plots
    plt.style.use("seaborn-darkgrid")
    _, [ax_roc, ax_det, ax_pr] = plt.subplots(1, 3, figsize=(20, 7))
    ax_roc.set_title("ROC curve")
    ax_det.set_title("DET curve")
    ax_pr.set_title("Precision-Recall curve")
    for classifier_name, y_pred_proba in classifiers.items():
        PrecisionRecallDisplay.from_predictions(
            y_true, y_pred_proba, name=classifier_name, ax=ax_pr
        )
        DetCurveDisplay.from_predictions(
            y_true, y_pred_proba, name=classifier_name, ax=ax_det
        )
        RocCurveDisplay.from_predictions(
            y_true, y_pred_proba, name=classifier_name, ax=ax_roc
        )
    plt.tight_layout()
