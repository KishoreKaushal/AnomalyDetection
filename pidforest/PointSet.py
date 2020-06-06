import numpy as np
from PIDTree import PIDTree


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