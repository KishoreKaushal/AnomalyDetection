from PIDTree import PIDTree
import numpy as np
import pandas as pd
import copy

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