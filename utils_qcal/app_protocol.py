# APP code
from mlquantify.model_selection import APP
import numpy as np
import pandas as pd

def APP_protocol_mlquantify(X, y, batch_size=100, n_prevalences=11, repeats=100, random_state=42):
    """
    Generates data batches with different class distributions using the APP method (mlquantify).

    Each batch contains a subset of the original data and the prevalence (proportion) of each
    class within that batch.

    Parameters
    ----------
    X : array-like
     - Feature matrix of the dataset.
    y : array-like
     - Label vector corresponding to X.
    batch_size : int, optional
     - Size of each generated batch. Default is 100.
    n_prevalences : int, optional
     - Number of different prevalences to generate. Default is 11.
    repeats : int, optional
     - Number of repetitions for each prevalence. Default is 100.
    random_state : int, optional
     - Seed for random number generation. Default is 42.

    Returns
    -------
    batches : list of tuples
        List containing all generated batches. Each element is a tuple:
        (batch_dataframe, prevalences), where:
            - batch_dataframe : pandas.DataFrame
                Subset of the batch samples with an additional 'target' column.
            - prevalences : list of floats
                Proportion of each class within the batch (list of length n_classes).

    Examples
    --------
    >>> batches = APP_protocol_mlquantify(X, y)
    >>> batch, prevalence = batches[0]
    """
    batches = []

    n_classes = len(np.unique(y))

    app = APP(batch_size=batch_size, n_prevalences=n_prevalences, repeats=repeats, random_state=random_state)
    for idx in app.split(X, y):
        X_batch = X[idx]
        y_batch = y[idx]

        # Converting the generated batch into a DataFrame
        batch = pd.DataFrame(X_batch)
        batch['target'] = y_batch

        # Proportion of the subgroups
        prevalence = [
                    batch[batch['target'] == i].shape[0] / len(batch)
                    for i in range(n_classes)
                ]

        batches.append((batch, prevalence))

    return batches