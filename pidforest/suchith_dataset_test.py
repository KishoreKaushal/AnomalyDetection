import pprint
import time

import pandas as pd
from PIDForest import PIDForest
import numpy as np
from sklearn.preprocessing import OrdinalEncoder



# train_data_path = "/home/kaushal/Documents/git/AnomalyDetectionDL/data/dataset.pkl"

train_data_path = "/home/kaushal/Downloads/dataset.pkl"

df = pd.read_pickle(train_data_path)

api_list = []
for i in range(0,14):
    for j in range(1,31):
        api_list.append("APIKEY{}".format(i*100 + j))


df = df[df['APIKEY'].isin(api_list)]
df['LABEL'] = df['LABEL'].apply(lambda x: 0 if x=='NOT ANAMOLY' else 1 )

cat = ['APIKEY', 'DAY', 'TIMEBIN']
cont = list(set(df.columns) - set(cat) - set(['ANAMOLYDISTNUM', 'NUMFAILURES']))


df_cat = df[cat]
df_cont = df[cont]

enc = OrdinalEncoder()
cat_transformed = enc.fit_transform(df_cat)

df_cat = pd.DataFrame(cat_transformed.astype(int), columns=df_cat.columns)

df = pd.DataFrame()
for col in df_cat.columns:
    df[col] = df_cat[col]

for col in df_cont.columns:
    df[col] = df_cont[col]

df = df.dropna()

# # rearrange label column to last
#
# l = list(set(df.columns) - set(['LABEL']))
# l.append('LABEL')
#
# df = df[l]


# apply pidforest


df_train = df[list(set(df.columns) - set(['LABEL']))]

# df_train.columns = list(range(len(df_train.columns)))
print("Is Nan {}".format(df_train.isnull().values.any()))
print("dfTrain shape : {}".format(df_train.shape))
print(df_train.head())

X_train = df_train.values

df_test = df.copy()
X_test = df_test.values

kwargs = {
'max_depth': 10,
'num_trees': 128,
'subsample_size': 128,
'max_buckets': 5,
'epsilon': 0.1,
'sample_axis' : 1,
'threshold': 0}

pprint.pprint("PIDforest parameters : ")
pprint.pprint(kwargs)

t0 = time.time()
forest = PIDForest(**kwargs).fit(df_train)
print("Training pidforest time: {}",format(time.time() - t0))

t0 = time.time()

df_score = forest.score(df_test, percentile=0.85)
print("Testing time on df_test.shape {}  : {}".format(df_test.shape, time.time() - t0))

df_test['score'] = df_score
df_test.sort_values('score', ascending=False, inplace=True)
df_test.to_csv('temp/suchith_result.csv')




