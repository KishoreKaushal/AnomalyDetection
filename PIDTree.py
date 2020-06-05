from PointSet import PointSet
from Histogram import Histogram
from Cube import Cube
from PIDForest import PIDForest
import pandas as pd


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

        self.density = -1
        self.child = []
        if (self.depth < self.forest.max_depth) and (len(self.point_set.df) > 1):
            self.find_split()


    def find_split(self):
        pass