"""Module for exploring the data."""

from collections import Counter
from typing import List, Optional, Tuple, Literal
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
PYLAB_PARAMS = {
    "legend.fontsize": 14,
    "figure.figsize": (7 * 1.6, 7),
    "axes.labelsize": 16,
    "axes.titlesize": 20,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
}

def plot_counts(
    data: List,
    titles: List[str],
    legend_title: str,
    suptitle: Optional[str] = None,
    kind: Literal["bar", "pie"] = "pie",
):
    """Create one plot for each data/title pair of kind `kind`."""
    plt.style.use("seaborn-darkgrid")
    pylab.rcParams.update(PYLAB_PARAMS)
    _, axes = plt.subplots(1, len(data), figsize=(20, 5), squeeze=False)
    axes = axes[0]
    legend = None

    if suptitle is not None:
        plt.suptitle(suptitle, fontsize=16)

    for axis, (dat, title) in zip(axes, zip(data, titles)):
        axis.set_title(title, fontweight="bold")

        if kind == "pie":
            counts = Counter(list(dat))
            legend = counts.keys() if legend is None else legend
            axis.pie(
                [counts[label] for label in legend],
                startangle=90,
                wedgeprops={"edgecolor": "black"},
                autopct="%1.f%%",
                explode=(0, 0.1),
                shadow=True,
                textprops={'fontsize': 18}
            )
            plt.legend(title=legend_title, labels=legend,loc="lower right")

        elif kind == "bar":
            ncount = len(dat)
            # pylint: disable=too-many-function-args
            sns.countplot(dat, ax=axis)
            for patch in axis.patches:
                axis.annotate(
                    f"{100.0*patch.get_height()/ncount:.2f}%",
                    (patch.get_x() + 0.1, patch.get_height() + 1),
                )

    plt.tight_layout()


def feature_distributions(
    data: pd.DataFrame,
    features: List[str],
    task: str,
    suptitle: Optional[str] = None,
    kind: Literal["hist", "violin"] = "violin",
):
    """Create a plot of type `kind` for each feature in `features`."""
    n_features = len(features)
    if n_features > 5:
        feature_distributions(
            data, features[: n_features // 2], task, suptitle, kind=kind
        )
        feature_distributions(data, features[n_features // 2 :], task, kind=kind)
    else:
        _, axes = plt.subplots(1, n_features, figsize=(20, 5), squeeze=False)
        axes = axes[0]
        plt.suptitle(suptitle, fontsize=16)

        for idx, feature in enumerate(features):
            axis = axes[idx]
            axis.set_title(feature)
            if kind == "violin":
                sns.violinplot(data=data[["Y", feature]], y=feature, x="Y", ax=axis)
                axis.set_xlabel(task)
                axis.set_ylabel(feature)
            elif kind == "hist":
                sns.histplot(
                    data=data.rename(
                        {"Y": task}, axis=1
                    ),
                    x=feature,
                    hue= task,
                    ax=axis,
                    bins=15,
                )

        plt.tight_layout()


def __plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


def plot_dendrogram(
    cor_matrix: pd.DataFrame,
    level: int = 6,
    color_threshold: float = 10,
    figsize: Tuple[int] = (25, 8),
):
    """Plot a Hierarchical Clustering Dendrogram."""

    cor_matrix_cleaned = cor_matrix.dropna(how="all").dropna(axis=1, how="all").abs()

    # setting distance_threshold=0 ensures we compute the full tree.
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    model = model.fit(cor_matrix_cleaned)

    plt.figure(figsize=figsize)
    plt.title("Hierarchical Clustering Dendrogram")
    # plot the top three levels of the dendrogram
    __plot_dendrogram(
        model=model,
        truncate_mode="level",
        p=level,
        color_threshold=color_threshold,
        labels=cor_matrix_cleaned.columns,
    )
    plt.xticks(fontsize=14)
    plt.xlabel("Number of points in node (or feature name)", fontsize=16)
    plt.tight_layout()
