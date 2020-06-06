import numpy as np
import pandas as pd
import random
import copy
from Histogram import Histogram

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


class Cube(object):

    def __init__(self, node, start, end):
        assert isinstance(node, PIDTree)

        self.node = node
        self.child = []
        self.start = start
        self.end = end
        self.num_attr = len(start)
        self.split_attr = None
        self.split_vals = []

        # calculating the log-volume of the subcube
        self.vol = 0
        for i in range(self.num_attr):
            self.vol += np.log(self.end[i] - self.start[i])


    def filter_df(self, df):
        assert isinstance(df, pd.DataFrame)

        df_filtered = df.copy()
        for col in df.columns:
            df_filtered = df_filtered[(self.start[col] <= df_filtered[col])
                                      & (df_filtered[col] <= self.end[col])]

        return df_filtered


    def split_df(self, df):
        num_child = len(self.child)

        # if no child nodes, we can't split
        if num_child == 0:
            return df

        splits = [[] for _ in range(num_child)]

        if df.shape[0] == 0:
            return splits

        # start and end of the interval for each split
        new_start = [copy.deepcopy(self.start) for _ in range(num_child)]
        new_end = [copy.deepcopy(self.end) for _ in range(num_child)]


        # new interval for first split
        # new_start[0][self.split_attr] = self.start[self.split_attr]
        new_end[0][self.split_attr] = self.split_vals[0]

        # new interval for last split
        new_start[-1][self.split_attr] = self.split_vals[-1]
        # new_end[-1][self.split_attr] = self.end[self.split_attr]

        # new interval for the ith split: 0 < i < num_child - 1
        for i in range(1, num_child-1):
            new_start[i][self.split_attr] = self.split_vals[i-1]
            new_end[i][self.split_attr] = self.split_vals[i]

        splits = [df[(new_start[i][self.split_attr] <= df[self.split_attr])
                    & (df[self.split_attr] <= new_end[i][self.split_attr])]
                    for i in range(num_child)]
        #
        #
        # splits[0] = df[(self.start[self.split_attr] <= df[self.split_attr])
        #                   & (df[self.split_attr] < self.split_vals[0])]
        #
        # splits[-1] = df[(self.split_vals[-1] <= df[self.split_attr])
        #                   & (df[self.split_attr] < self.end[self.split_attr])]
        #
        # for k in range(1, num_child - 1):
        #     splits[k] = df[(self.split_vals[k-1] <= df[self.split_attr])
        #                   & (df[self.split_attr] < self.split_vals[k])]

        return (splits, new_start, new_end)


class PIDTree(object):

    def __init__(self, **kwargs):
        self.depth = kwargs['depth']
        self.forest = kwargs['forest']

        assert isinstance(self.forest, PIDForest)
        assert isinstance(kwargs['df'], pd.DataFrame)

        if self.depth == 0:
            self.node_id = [0]
            self.cube = Cube(node=self, start=self.forest.start, end=self.forest.end)
            self.point_set = PointSet(node=self, df=kwargs['df'])
        else:
            self.node_id = kwargs['id']
            self.cube = Cube(node=self, start=kwargs['start'], end=kwargs['end'])
            self.point_set = PointSet(node=self, df=self.cube.filter_df(kwargs['df']))

        self.sparsity = -1
        self.child = []
        if (self.depth < self.forest.max_depth) and (len(self.point_set.df) > 1):
            self.find_split()
        else: # compute sparsity of the leaf node
            self.compute_sparsity()


    def find_split(self):
        col_with_variance = [col for col in self.point_set.df.columns if len(self.point_set.val[col]) > 1]

        if len(col_with_variance) == 0:
            return

        buckets = {}
        var_red = {}
        for col in col_with_variance:
            hist = Histogram(arr=self.point_set.gap[col] / self.point_set.count[col],
                             count=self.point_set.count[col],
                             max_buckets=self.forest.max_buckets,
                             eps=self.forest.epsilon)

            _, var_red[col], buckets[col] = hist.best_split()

        if np.max(list(var_red.values())) <= self.forest.threshold:
            return

        split_attr = np.random.choice(col_with_variance,
                            p=list(var_red.values()) / np.sum(list(var_red.values())))

        self.cube.split_attr = split_attr
        self.cube.split_vals = [(self.point_set.val[split_attr][i - 1] + self.point_set.val[split_attr][i]) / 2
                                for i in buckets[split_attr]]

        splits_df, new_start, new_end = self.cube.split_df(df=self.point_set.df)

        for i in range(len(splits_df)):
            new_id = copy.deepcopy(self.node_id)
            new_id.append(i)
            kwargs = {
                'depth' : self.depth + 1,
                'forest' : self.forest,
                'start' : new_start[i],
                'end' : new_end[i],
                'df' : splits_df[i],
                'id' : new_id
            }
            child_node = PIDTree(**kwargs)
            self.child.append(child_node)
            self.cube.child.append(child_node.cube)

    def compute_sparsity(self):
        if len(self.child) == 0:
            self.sparsity = self.cube.vol - np.log(self.point_set.df.shape[0])
        else: # sparsity not matters for internal nodes
            self.sparsity = -1

    def set_score(self, df, score_col_name):
        if self.depth == 0:
            df = self.cube.filter_df(df)

        if len(self.child) != 0:
            split_df = self.cube.split_df(df)
            for i in range(len(split_df)):
                self.child[i].set_score(split_df[i])
        else:
            df[score_col_name] = (-1 / self.sparsity)


class PointSet(object):

    def __init__(self, node, df):
        self.node = node
        self.df = df
        self.val = {}
        self.count = {}
        self.gap = {}

        for col in range(self.df.columns):
            val, count = np.unique(self.df[col], return_counts=True)
            self.val[col] = val
            self.count[col] = count

            if len(val) <= 1:
                gap = [0]
            else:
                # sum of all gap for a col is equal to start[col] - end[col]
                gap = np.zeros(len(val))
                gap[0] = (val[0] + val[1]) / 2 - self.node.cube.start[col]
                gap[-1] = self.node.cube.end[col] - (val[-1] + val[-2]) / 2
                for i in range(1, len(val) - 1):
                    gap[i] = (val[i + 1] - val[i - 1]) / 2
            self.gap[col] = gap
