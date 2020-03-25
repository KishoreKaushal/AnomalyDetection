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
    """

    def __init__(self, num_trees: int = 128, subsample_size: int = 256,
                 copy_x: bool = True):
        self.num_trees = num_trees
        self.subsample_size = subsample_size
        self.copy_x = copy_x

    def fit(self, ):
        """ Fit the  """
        pass
