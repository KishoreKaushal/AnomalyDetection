import argparse
from gensim import models
import pandas as pd
import numpy as np
import pprint
import multiprocessing
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from os import path

MAX_CORES = 6


def parse_args():
    parser = argparse.ArgumentParser(add_help=True, description='testing cat2vec')
    parser.add_argument('-j', '--jobs', type=int, default=-1, help='number of worker jobs')
    parser.add_argument('-e', '--embdsize', type=int, default=4, help='size of embd vector')
    parser.add_argument('-b', '--numbins', type=int, default=10, help='number of bins for '
                                                                      'converting continuos data to categoical data')
    required_named = parser.add_argument_group('required named arguments')
    required_named.add_argument('-i', '--input', type=str, help='path of the input pickle file', required=True)
    required_named.add_argument('-f', '--catfeat', type=str, help='features comma separated')
    return parser.parse_args()


def vector_embedding(df, cat_feat, cat2vec_dict, drop=True):
    df = df.copy()
    # prepare the columns for the vector embedding
    # for f in cat_feat:
    #     for i in range(embd_size):
    #         df["{}_{}".format(f, i)] = 0.0

    # insert the embedding for all the features
    for f in cat_feat:
        for fv, embd in cat2vec_dict[f].items():
            for i in range(len(embd)):
                df.loc[df[f] == fv, "{}_{}".format(f, i)] = embd[i]

    # once inserted drop the original columns
    if drop:
        return df.drop(columns=cat_feat)

    return df


def cat2vec(df, cat_list, num_bins=10, num_cores=-1, embd_size = 4):
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

    if num_cores <= 0:
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
    cat2vec_dict = cat2vec(df, cat_list=cat_feature_list,
                           num_bins=args.numbins,
                           embd_size=args.embdsize,
                           num_cores=args.jobs)

    with open(path.join('temp', path.basename(args.input)+'_embd.pkl'), 'wb') as handle:
        pickle.dump(cat2vec_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # printing the distance matrix for each features
    global_vec_list = []
    global_vec_ord = []
    for f in cat_feature_list:
        val_ord = []
        vec_list = []
        dist_mat = None
        for fv, embd in cat2vec_dict[f].items():
            vec_list.append(embd)
            val_ord.append(fv)
            global_vec_ord.append(str(f) + ":" + str(fv))

        global_vec_list.extend(vec_list)

        dist_mat = distance_matrix(vec_list, vec_list)

        pp.pprint("----- Computing Distance matrix -----")
        pp.pprint("Embeddings of the feature `{}`".format(f))
        pp.pprint(cat2vec_dict[f])
        pp.pprint("Unique value of feature `{}`".format(f))
        pp.pprint(val_ord)
        pp.pprint("Distance matrix:")
        pp.pprint(np.round(dist_mat,2))

    # printing the complete distance matrix.
    dist_mat = distance_matrix(global_vec_list, global_vec_list)
    pp.pprint("----- Computing Distance matrix (inter features) -----")
    pp.pprint("Order of the embeddings: ")
    pp.pprint(global_vec_ord)
    pp.pprint("Distance matrix:")
    pp.pprint(np.round(dist_mat, 2))
    sns.heatmap(dist_mat, annot=True,
                xticklabels=global_vec_ord,
                yticklabels=global_vec_ord,
                fmt='.2f',
                cbar=False)
    plt.savefig(path.join('temp', path.basename(args.input)+'_dist_mat.png'))
    plt.show()

    # save the dataframe to pickle file
    df_new = vector_embedding(df, cat_feature_list, args.embdsize, cat2vec_dict)
    df_new.to_pickle(path.join('data', path.basename(args.input)+'_embd.pkl'))

