from PointSet import PointSet
from Histogram import Histogram
from Cube import Cube
from PIDForest import PIDForest
import pandas as pd
import numpy as np
import copy


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