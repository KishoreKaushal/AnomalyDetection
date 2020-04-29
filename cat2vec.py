import argparse
from gensim import models
import pandas as pd
import numpy as np
import pprint
import multiprocessing
from scipy.spatial import distance_matrix


MAX_CORES = 6


def parse_args():
    parser = argparse.ArgumentParser(add_help=True, description='testing cat2vec')
    required_named = parser.add_argument_group('required named arguments')
    required_named.add_argument('-i', '--input', type=str, help='path of the input pickle file', required=True)
    required_named.add_argument('-f', '--catfeat', type=str, help='features comma separated')
    return parser.parse_args()


def cat2vec(df, cat_list, num_bins=10, num_cores='auto', embd_size = 4):
    """
    Returns a dictionary containing the vector embedding of
    all unique values of the categorical features.

    Parameters
    ----------
    df : pd.DataFrame (n_samples, n_features)
        Tabular data.

    num_bins : int, optional, default 10
        Used to preprocess the continuous data to
        convert to categorical data for word2vec model.

    num_cores : 'auto' or int, optional, default 'auto'
        Specify number of parallel jobs for word2vec model.
        Default value 'auto' selects it according to following rule:
        num_cores = min (MAX_CORES, TOTAL_CORES_AVAILABLE)

    embd_size : int, optional, default 4
        Specify the size of the embedding vector.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError('df should be of type pd.DataFrame')

    new_df = pd.DataFrame()

    for col in df.columns:
        if col in cat_list: # categorical data
            new_df[col] = col + '-cat-' + df[col].astype(str)
        else: # non-categorical data
            new_df[col] = col + '-bin-' + pd.cut(df[col], num_bins, labels=False).astype(str)

    sentence_list = new_df.values.tolist()

    if num_cores == 'auto':
        num_cores = min(MAX_CORES, multiprocessing.cpu_count())

    model = models.Word2Vec(sentence_list, min_count=1,
                                     size=embd_size, workers=num_cores)

    cat2vec_dict = dict()
    for cat in cat_list:
        cat2vec_dict[cat] = dict()
        for item in df[cat].unique():
            cat2vec_dict[cat][item] = model.wv[str(cat)+'-cat-'+str(item)]

    return cat2vec_dict

if __name__ == "__main__":
    args = parse_args()
    df = pd.read_pickle(args.input)

    cat_feature_list = [x.strip() for x in args.catfeat.split(',')]

    pp = pprint.PrettyPrinter(indent=4)
    cat2vec_dict = cat2vec(df, cat_list=cat_feature_list, num_bins=10, embd_size=4)

    # printing the distance matrix for each features
    for f in cat_feature_list:

        val_ord = []
        vec_list = []
        dist_mat = None
        for fv, embd in cat2vec_dict[f].items():
            vec_list.append(embd)
            val_ord.append(fv)

        dist_mat = distance_matrix(vec_list, vec_list)

        pp.pprint("----- Computing Distance matrix -----")
        pp.pprint("Embeddings of the feature `{}`".format(f))
        pp.pprint(cat2vec_dict[f])
        pp.pprint("Unique value of feature `{}`".format(f))
        pp.pprint(val_ord)
        pp.pprint("Distance matrix:")
        pp.pprint(np.round(dist_mat,2))
