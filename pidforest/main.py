from PIDForest import PIDForest
import numpy as np
import pandas as pd

# train_data_path = "/home/kaushal/Documents/AnomalyDetection/Data/cat2vec_train.pickle"
# test_data_path = "/home/kaushal/Documents/AnomalyDetection/Data/cat2vec_test.pickle"

train_data_path = "/home/kaushal/Documents/git/AnomalyDetection/data/pima.pickle"
test_data_path = "/home/kaushal/Documents/git/AnomalyDetection/data/pima.pickle"


df_train = pd.read_pickle(train_data_path)
df_train = df_train[df_train.columns[:-1]]

df_test = pd.read_pickle(test_data_path)
X_train = df_train.values
X_test = df_test.values

kwargs = {
'max_depth': 10,
'num_trees': 12,
'subsample_size': 64,
'max_buckets': 5,
'epsilon': 0.1,
'sample_axis' : 1,
'threshold': 0}

forest = PIDForest(**kwargs).fit(df_train)
df_score = forest.score(df_test, percentile=0.80)
df_test['score'] = df_score
df_test.sort_values('score', ascending=False, inplace=True)
df_test.to_csv('temp/result.csv')
