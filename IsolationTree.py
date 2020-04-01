import random
import utils

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
        self.parent = None
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

    def path_length(self, df_inst, hlim, n):
        """
        Parameters:
        -----------
        df_inst : pd.DataFrame
            Collection on instances whose path length has to calculated.

        hlim : numeric value
            Height Limit.

        n : numeric value
            Dataset size for estimating the size given the sample size.
        """
        path_length_arr = []

        # this can be parallelized -- using map function
        for _, inst in df_inst.iterrows():
            # this function will only be executed through root node
            if self.root == False:
                raise PermissionError("Only root node is permitted to execute this function.")

            curr_node = self
            plen = 0.0
            # better not to write this with recursion
            while(True):
                if (curr_node.left != None and curr_node.right != None) or curr_node.depth > hlim:
                    plen += utils.c(curr_node.size, n)
                    break
                else:
                    plen += 1
                    if inst[curr_node.splitting_attr] < curr_node.splitting_val:
                        curr_node = curr_node.left
                    else:
                        curr_node = curr_node.right

            path_length_arr.append(plen)

        return path_length_arr


