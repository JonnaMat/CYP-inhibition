"""Module for exploring the data."""

from collections import Counter
from typing import List, Literal, Optional
from math import ceil
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_counts(
    data: List,
    titles: List[str],
    legend_title: str,
    suptitle: Optional[str] = None,
    kind: Literal["bar", "pie"] = "pie",
    legend_labels: Optional[List] = None,
):
    """Create one plot for each data/title pair of kind `kind`."""
    _, axes = plt.subplots(1, len(data), figsize=(20, 5), squeeze=False)
    axes = axes[0]

    plt.suptitle(suptitle, fontsize=16)

    for axis, (dat, title) in zip(axes, zip(data, titles)):
        axis.set_title(title)

        if kind == "pie":
            counts = Counter(list(dat))
            legend_labels = counts.keys() if legend_labels is None else legend_labels
            axis.pie(
                [counts[label] for label in legend_labels],
                startangle=90,
                wedgeprops={"edgecolor": "black"},
                autopct="%1.f%%",
                explode=(0, 0.1),
                shadow=True,
            )
            plt.legend(title=legend_title, labels=legend_labels, loc="lower right")

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
    suptitle: Optional[str] = None,
):
    """Create one violin plot for each feature in `features`."""
    n_features = len(features)
    if n_features > 5:
        feature_distributions(
            data,
            features[:n_features//2],
            suptitle,
        )
        feature_distributions(
            data,
            features[n_features//2:],
        )
    else:
        _, axes = plt.subplots(
            1, n_features, figsize=(20, 5), squeeze=False
        )
        axes = axes[0]

        plt.suptitle(suptitle, fontsize=16)

        for idx, feature in enumerate(features):
            axis = axes[idx]
            axis.set_title(feature)

            sns.violinplot(data=data[["Y", feature]], y=feature, x="Y", ax=axis)
            axis.set_xlabel("CYP inhibition")
            axis.set_ylabel(feature)

        plt.tight_layout()
