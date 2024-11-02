import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def generate_points(mean: float = 0, sd: float = 1, size: int = 100):
    """_summary_

    Args:
        mean (float, optional): mean. Defaults to 0.
        sd (float, optional): standard deviation. Defaults to 1.
        size (int, optional): number of points to generate. Defaults to 100.

    Returns:
        np.array: array of points
    """
    return np.random.normal(mean, sd, size)


# Generate clusters
def generate_clusters(
    k: int = 3, sd: int = 1, size: int = 100, loc_range: tuple = (-10, 10)
):
    """Generates K clusters within a given range.

    Args:
        k (int, optional): number of clusters. Defaults to 3.
        sd (int, optional): the standard deviation. Defaults to 1.
        size (int, optional): size of the cluster. Defaults to 100.
        range (tuple, optional): x and y range of the mean. Defaults to (-10, 10).

    Returns:
        pandas.Dataframe: dataframe of the clusters.
    """
    X = np.array([])
    y = np.array([])
    labels = np.array([])
    for i in range(k):
        x_mean = np.random.randint(loc_range[0], loc_range[1])
        y_mean = np.random.randint(loc_range[0], loc_range[1])
        X = np.concatenate((X, generate_points(x_mean, sd, size)))
        y = np.concatenate((y, generate_points(y_mean, sd, size)))
        labels = np.concatenate((labels, np.full((size,), i, dtype=np.uint16)), axis=0)

    return pd.DataFrame({"x": X, "y": y, "label": labels})

def plot(df, labels):
    plot = plt.scatter(x=df.x, y=df.y, c=df.label, cmap=cm.jet, alpha=0.5)
    plt.legend(handles=plot.legend_elements()[0], labels=list(labels))
