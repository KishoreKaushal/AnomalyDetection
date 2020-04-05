from numpy import euler_gamma
from numpy import log

def avg_path_len_given_sample_size(sample_size, n) -> float:
    """"
    Average of path length given subsample size.

    Arguments:
    ----------
    sample_size : float

    n : int
        Size of the complete dataset for the forest.
    """

    if sample_size > 2:
        return 2 * (log(sample_size-1) + euler_gamma) - 2 * (sample_size - 1)/n
    elif sample_size == 2:
        return 1.0
    else:
        return 0.0

def auc(df_y, score):
    """
    Return AUC.

    Arguments:
    ----------
    df_y: DataFrame
        DataFrame object where each instance is 1 for anomaly, 0 for normal class.

    score: array of floats
        Anomaly scores.

    Return
    ------
        AUC score.
    """

    df_y = df_y.rename(columns={df_y.columns[0]: 'is_anomaly'})
    num_anomalies = df_y[df_y['is_anomaly'] == 1].sum()[0]
    num_normal_points = df_y.shape[0] - num_anomalies
    df_y['score'] = score
    df = df_y.sort_values(by='score', ascending=False)
    df = df.reset_index(drop=True)
    df.index = df.index + 1
    df = df.reset_index()
    s = df[df['is_anomaly'] == 1]['index'].sum()

    return  (s - (num_anomalies**2 + num_anomalies)/2) / (num_anomalies * num_normal_points)