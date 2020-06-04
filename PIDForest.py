import numpy as np
import pandas as pd
from random import sample
from PIDTree import PIDTree

class PIDForest(object):

    def __init__(self, copy_x : bool = True, **kwargs):
        self.num_trees = kwargs['num_trees']
        self.max_depth = kwargs['max_depth']
        self.subsample_size = kwargs['subsample_size']
        self.max_buckets = kwargs['max_buckets']
        self.epsilon = kwargs['epsilon']
        self.sample_axis = kwargs['sample_axis']
        self.threshold = kwargs['threshold']
        self.copy_x = copy_x
        self.n_leaves = np.zeros(self.num_trees)
        self.pid_trees = None


    def fit(self, df):


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
        subsamples = np.array(subsamples).reshape((self.num_trees, self.subsample_size))

        # TODO - complete this code to create PIDTrees
        self.pid_trees = [_ for subsample in subsamples]

        return self



