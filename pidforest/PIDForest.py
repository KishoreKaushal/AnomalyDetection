import numpy as np
import pandas as pd
import random
from PIDTree import PIDTree

DELTA = 1e-3

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
        self.num_attr = None
        self.num_inst = None
        self.start = dict()
        self.end = dict()


    def fit(self, df):

        self.df = df

        if not isinstance(self.df, pd.DataFrame):
            self.df = pd.DataFrame(data=self.df)

        self.num_inst, self.num_attr = self.df.shape

        if self.subsample_size * self.num_trees > self.num_inst:
            raise ValueError("subsample_size ({}) * num_trees ({}) is greater than "
                             "number of instances ({}) in the dataset."
                             .format(self.subsample_size, self.num_trees, self.num_inst))

        subsamples = random.sample(range(self.num_inst), self.subsample_size * self.num_trees)

        # partitioning into self.num_trees equal chunks of size self.subsample_size
        subsamples = np.array(subsamples).reshape((self.num_trees, self.subsample_size))


        # setting the initial interval
        for col in df.columns:
            if len(pd.unique(df[col])) <= 1:
                raise ValueError("No entropy in the column: ", col)
            self.start[col], self.end[col] = (df[col].min() - DELTA, df[col].max() + DELTA)

        self.pid_trees = []
        for i in range(len(subsamples)):
            kwargs = {
                'depth': 0,
                'forest': self,
                'start': self.start,
                'end': self.end,
                'df': self.df.iloc[subsamples[i]],
                'id': [i, 0]
            }
            self.pid_trees.append(PIDTree(**kwargs))

        return self

    def score(self, df, percentile=85):
        assert isinstance(df, pd.DataFrame)

        df = df.copy()
        tree_score_col = []
        for i in range(self.num_trees):
            new_col_name = 'anomaly_score_tree_' + str(i)
            tree_score_col.append(new_col_name)
            df[new_col_name] = 0.0
            self.pid_trees[i].set_pid_score(df, score_col_name=new_col_name)

        df['anomaly_score'] = df[tree_score_col].quantile(q=percentile, axis=1)

        return df['anomaly_score']



