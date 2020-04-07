from FeedbackIsolationTree import FeedbackIsolationTree
import pandas as pd
from copy import deepcopy
import random
import numpy as np
import utils

LOSS_FN_TYPES = frozenset(['linear', 'log-likelihood'])

class FeedbackIsolationForest(object):
    """
        Feedback Guided Isolation Forest

        Parameters
        ----------
        num_trees : int, optional, default 128
            Number of isolation trees in the forest.

        subsample_size : int, optional, default 256
            Sampling size for the dataset for isolation trees.

        copy_x : bool, optional, default True
            If True, dataset x will be copied; else, it may be overwritten.

        lrate: float, optional, default 1.0
            Learning rate for mirror descent algorithm.

        df : array-like of shape (n_samples, n_features)
            Training data.

        Attributes
        ----------

        feedback_isolation_trees : list, default None
            List of IsolationTree.
        """

    def __init__(self, num_trees: int = 128, subsample_size: int = 256,
                 copy_x: bool = True, lrate : float = 1.0, df = None):
        self.num_trees = num_trees
        self.subsample_size = subsample_size
        self.copy_x = copy_x
        self.df = None
        self.feedback_isolation_trees = None
        self.lrate = lrate
        
        if df is not None:
            self.init_base(df)

    def init_base(self, df):
        """
        Initializing the base isolation trees.

        Parameters
        ----------
        df : pd.Dataframe object
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

        subsamples = random.sample(range(num_inst), self.subsample_size * self.num_trees)

        # partitioning into self.num_trees equal chunks of size self.subsample_size
        subsamples = np.array(subsamples).reshape(shape=(self.num_trees, self.subsample_size))

        # creating isolation trees -- this code can be parallelized -- using map function
        self.feedback_isolation_trees = [FeedbackIsolationTree().fit(df.loc[subsample])
                                for subsample in subsamples]

        return self


    def update_weights(self, hlim, feedback, lrate, inst):
        """
        Parameters:
        -----------
        hlim : int
            Height (or depth) limit of the tree.

        inst : pd.DataFrame
            An instance which is used to compute loss.

        lrate : numeric value
            Learning rate of the mirror descent algorithm.

        feedback : {+1, -1}
            +1 means that it is the inst is anomaly otherwise not.
        """

        # this can be parallelized using -- map function
        for ftree in self.feedback_isolation_trees:
            ftree.update_weights(hlim, feedback, lrate, inst)

        return self

    def score(self, df_inst):
        """
        Return an array of scores of instances.

        Parameters
        ----------
        df_inst : pd.DataFrame object
            One or more instances.

        Return
        ------
        scores : numpy.array
            An array of anomaly scores for each instance in df_inst.
        """

        if not isinstance(df_inst, pd.DataFrame):
            df_inst = pd.DataFrame(data=df_inst)

        # this can be parallelized -- using map function or using joblib
        # scores here is a 2D array of shape => (num_trees, num_df_inst)
        path_lengths = np.array([tree.path_length(df_inst)
                                 for tree in self.feedback_isolation_trees])

        # scores store sum of path length of an instance across all trees
        # for each instance in df_inst. Shape of scores => (num_df_inst,)
        scores = -1 * np.sum(path_lengths, axis=0)

        return scores
