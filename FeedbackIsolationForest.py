import IsolationForest

LOSS_FN_TYPES = frozenset(['linear', 'log-likelihood'])

class FeedbackIsolationForest(IsolationForest):
    """
        Feedback Guided Isolation Forest

        Parameters
        ----------
        num_trees : int, optional, default 128
            Number of isolation trees in the forest. Inherited.

        subsample_size : int, optional, default 256
            Sampling size for the dataset for isolation trees. Inherited.

        copy_x : bool, optional, default True
            If True, dataset x will be copied; else, it may be overwritten. Inherited.

        loss_fn: str, optional, default linear
            Loss function to use with mirror descent algorithm.

        set_member: function-type, default returns True for all values
            A function which is used to check set membership of the weights in the desired set.
            For a tree, it should be (lambda wt => (wt >= 0)).

        lrate: float, optional, default 1.0
            Learning rate for mirror descent algorithm.

        Attributes
        ----------

        isolation_trees : list, default None
            List of IsolationTree. Inherited.

        df : array-like of shape (n_samples, n_features)
            Training data. Inherited.

        """

    def __init__(self, num_trees: int = 128, subsample_size: int = 256, copy_x: bool = True,
                 loss_fn : str = 'linear', lrate : float = 1.0, set_member = (lambda wt: (wt >= 0))):

        if loss_fn not in LOSS_FN_TYPES:
            raise ValueError('{} loss not supported'.format(loss_fn))

        super(FeedbackIsolationForest, self).__init__(num_trees, subsample_size, copy_x)
        self.loss_fn = loss_fn
        self.set_member = set_member
        self.lrate = lrate


    def update_weights(self):

        pass

    def score(self):
        pass
