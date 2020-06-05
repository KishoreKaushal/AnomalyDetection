from PIDTree import PIDTree
import numpy as np

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


