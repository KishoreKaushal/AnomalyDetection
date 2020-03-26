import random

class IsolationTree(object):
    """
    Isolation Tree

    Attributes
    ----------
    right : IsolationTree
        Right child of the node.

    left : IsolationTree
        Left child of the node.

    parent : IsolationTree
        Parent of the current node.

    depth : int
        Shortest distance of the current node
        from the root of the tree.

    df : DataFrame
        Data stored in the current node. In the context of the Isolation Forest
        this will just store reference to the dataset for this node.

    root : bool, default False
        For root node in the tree this is set True.

    size : int
        Number of instances in the dataset.

    weight : float
        Weight of the edge between parent node and self. For root node of the
        Isolation Tree this value will be 0.

    splittingAttr : object
        Identifier of the attribute which is used to split the input dataset.

    splittingVal :  float
        Split value of the attribute.

    splittingAttrRange : (minAttrVal: float, maxAttrVal: float)
        Range of splitting attribute.
        
    """

    def __init__(self):
        self.right = None
        self.left = None
        self.parent = None
        self.df = None
        self.root = False

        self.depth = None
        self.size = None
        self.weight = 1

        self.splitting_attr = None
        self.splitting_val = None
        self.splitting_attr_range = None

    def __str__(self) -> str:
        return "complete this code"

    def fit(self, df):
        # to distinguish root node
        self.root = True
        self.weight = 0
        self.init_tree(df, depth=0)
        return self

    def init_tree(self, df, depth):
        num_inst, num_attr = df.shape
        self.depth = depth
        self.size = num_inst
        self.df = df

        # if df can be splitted
        if self.size > 1:
            self.splitting_attr = random.randint(0, num_attr-1)
            self.splitting_attr_range = [df[self.splitting_attr].min(),
                                         df[self.splitting_attr].max()]

            self.splitting_val = random.uniform(self.splitting_attr_range[0],
                                                self.splitting_attr_range[1])

            df_left = df[df[self.splitting_attr] < self.splitting_val]
            df_right = df[df[self.splitting_attr] >= self.splitting_val]

            self.left = IsolationTree().init_tree(df_left, depth=depth + 1)
            self.right = IsolationTree().init_tree(df_right, depth=depth + 1)

            self.left.parent = self
            self.right.parent = self

        return self
