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

    data : object
        Data stored in the current node. In the context of the Isolation Forest
        this will just store the indexes of the dataset in the node.

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
        self.data = None

        self.depth = None
        self.size = None
        self.weight = None

        self.splittingAttr = None
        self.splittingVal = None
        self.splittingAttrRange = None

    def __str__(self) -> str:
        return "complete this code"


