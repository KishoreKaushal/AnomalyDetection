import IsolationForest


class FeedbackIsolationForest(IsolationForest):
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

        Attributes
        ----------

        isolation_trees : list, default None
            List of IsolationTree

        df : array-like of shape (n_samples, n_features)
            Training data.

        """

    def __init__(self, num_trees: int = 128, subsample_size: int = 256,
                 copy_x: bool = True):
        super(FeedbackIsolationForest, self).__init__(num_trees, subsample_size, copy_x)

    def update_weights(self):
        pass

    def score(self):
        pass
