from PIDTree import PIDTree
import numpy as np
import pandas as pd

class Cube(object):

    def __init__(self, node, start, end):
        assert isinstance(node, PIDTree)

        self.node = node
        self.child = []
        self.start = start
        self.end = end
        self.num_attr = len(start)
        self.split_axis = -1
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