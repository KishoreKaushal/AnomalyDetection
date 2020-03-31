import FeedbackIsolationTree

LOSS_FN_TYPES = frozenset(['linear', 'log-likelihood'])

class FeedbackIsolationForest(object):
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

        self.num_trees = num_trees
        self.subsample_size = subsample_size
        self.copy_x = copy_x
        self.df = None
        self.feedback_isolation_trees = None

        self.loss_fn = loss_fn
        self.set_member = set_member
        self.lrate = lrate

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
        
        pass
