from gensim import models
import pandas as pd
import numpy as np
import multiprocessing

MAX_CORES = 6

def cat2vec(df, cat_list, num_bins=10, num_cores='auto', embd_size = 4):
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
    pass