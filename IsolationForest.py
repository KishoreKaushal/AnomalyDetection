from copy import deepcopy
import pandas as pd
from random import sample
import numpy as np
import IsolationTree

class IsolationForest(object):
    """
    Isolation Forest

    Parameters
    ----------
    num_trees : int, optional, default 128
        Number of isolation trees in the forest.

    subsample_size : int, optional, default 256
        Sampling size for the dataset for isolation trees.

    copy_x : bool, optional, default True
        If True, dataset x will be copied; else, it may be overwritten.

    Attributes
    ----------

    isolation_trees : list, default None
        List of IsolationTree

    df : array-like of shape (n_samples, n_features)
        Training data.

    """

    def __init__(self, num_trees: int = 128, subsample_size: int = 256,
                 copy_x: bool = True):
        self.num_trees = num_trees
        self.subsample_size = subsample_size
        self.copy_x = copy_x
        self.df = None
        self.isolation_trees = None

    def fit(self, df):
        """
        Fit the Isolation Forest model on the dataset

        Parameters
        ----------
        df : pd.DataFrame object
        """
        if self.copy_x:
            self.df = deepcopy(df)
        else:
            self.df = df

        if not isinstance(self.df, pd.DataFrame):
            self.df = pd.DataFrame(data=self.df)

        num_inst, num_attr = self.df.shape

        if self.subsample_size * self.num_trees > num_inst:
            raise ValueError("subsample_size ({}) * num_trees ({}) is greater than "
                             "number of instances ({}) in the dataset."
                             .format(self.subsample_size, self.num_trees, num_inst))

        subsamples = sample(range(num_inst), self.subsample_size * self.num_trees)

        # partitioning into self.num_trees equal chunks of size self.subsample_size
        subsamples = np.array(subsamples).reshape(shape=(self.num_trees, self.subsample_size))

        # creating isolation trees -- this code can be parallelized
        self.isolation_trees = [IsolationTree().fit(df.loc[subsample])
                                for subsample in subsamples]

        return self