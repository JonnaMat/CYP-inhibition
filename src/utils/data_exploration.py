"""Module for exploring the data."""

from collections import Counter
from typing import List, Literal, Optional
import matplotlib.pyplot as plt
import seaborn as sns


def plot_counts(
    data: List,
    titles: List[str],
    legend_title: str,
    suptitle: Optional[str] = None,
    kind: Literal["bar", "pie"] = "bar",
    legend_labels: Optional[List] = None,
):
    """Creates one plot for each data/title pair of kind `kind`."""
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
