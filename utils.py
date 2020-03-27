from numpy import euler_gamma
from numpy import log



def c(sample_size, n) -> float:
    """"
    Average of path length given subsample size.

    Arguments:
    ----------
    sample_size : float

    n : int
        Size of the complete dataset for the forest.
    """

    if sample_size > 2:
        2 * (log(sample_size-1) + euler_gamma) - 2 * (sample_size - 1)/n
    elif sample_size == 2:
        return 1.0
    else:
        return 0.0