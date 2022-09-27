"""Module for exploring the data."""

from typing import List
import matplotlib.pyplot as plt
import seaborn as sns

# pylint: disable=too-many-function-args


def plot_counts(data: List, titles: List[str], suptitle=None):
    """Creates one bar plot for each data/title pair"""
    _, axes = plt.subplots(1, len(data), figsize=(20, 5))
    plt.suptitle(suptitle, fontsize=16)
    if len(titles) == 1:
        axes.set_title(titles[0])
        ncount = len(data[0])
        sns.countplot(data[0], ax=axes)
        for p in axes.patches:
            axes.annotate(
                f"{100.0*p.get_height()/ncount:.2f}%",
                (p.get_x() + 0.1, p.get_height() + 1),
            )
    else:
        for ax, (d, title) in zip(axes, zip(data, titles)):
            ax.set_title(title)
            ncount = len(d)
            sns.countplot(d, ax=ax)
            for p in ax.patches:
                ax.annotate(
                    f"{100.0*p.get_height()/ncount:.2f}%",
                    (p.get_x() + 0.1, p.get_height() + 1),
                )

    plt.tight_layout()
