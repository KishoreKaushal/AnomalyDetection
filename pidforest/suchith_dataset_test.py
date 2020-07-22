import pprint
import time
import pandas as pd
from PIDForest import PIDForest, save_pidforest_model
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_auc_score



train_data_path = "~/dataset/train_19072020.pkl"
test_data_path = "~/dataset/test_19072020.pkl"

df_train = pd.read_pickle(train_data_path)
df_test = pd.read_pickle(test_data_path)

# api_list = []
# for i in range(0,14):
#     for j in range(1,31):
#         api_list.append("APIKEY{}".format(i*100 + j))
#
#
# df = df[df['APIKEY'].isin(api_list)]


df_train['LABEL'] = df_train['LABEL'].apply(lambda x: 0 if x=='NOT ANAMOLY' else 1 )
df_test['LABEL'] = df_test['LABEL'].apply(lambda x: 0 if x=='NOT ANAMOLY' else 1 )

cat = ['APIKEY', 'DAY', 'TIMEBIN']
cont = list(set(df_train.columns) - set(cat) - set(['ANAMOLYDISTNUM']))


## transformations of df_train
df_cat_train = df_train[cat]
df_cont_train = df_train[cont]

enc = OrdinalEncoder()
cat_transformed = enc.fit_transform(df_cat_train)
df_cat_train = pd.DataFrame(cat_transformed.astype(int), columns=df_cat_train.columns)

df = pd.DataFrame()
for col in df_cat_train.columns:
    df[col] = df_cat_train[col]

for col in df_cont_train.columns:
    df[col] = df_cont_train[col]

df = df.dropna()
df_train = df[list(set(df.columns) - set(['LABEL']))]

## transformations of df_test
df_cat_test = df_test[cat]
df_cont_test = df_test[cont]

cat_transformed = enc.transform(df_cat_test)
df_cat_test = pd.DataFrame(cat_transformed.astype(int), columns=df_cat_test.columns)

df = pd.DataFrame()
for col in df_cat_test.columns:
    df[col] = df_cat_test[col]

for col in df_cont_test.columns:
    df[col] = df_cont_test[col]

df = df.dropna()
df_test = df


# # rearrange label column to last
#
# l = list(set(df.columns) - set(['LABEL']))
# l.append('LABEL')
#
# df = df[l]


# apply pidforest


# df_train.columns = list(range(len(df_train.columns)))
print("Is Nan {}".format(df_train.isnull().values.any()))
print("dfTrain shape : {}".format(df_train.shape))
print(df_train.head())

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

# save the model
# save_pidforest_model(forest, '/home/kaushal/temp/')

t0 = time.time()

df_score = forest.score(df_test, percentile=0.85)
print("Testing time on df_test.shape {}  : {}".format(df_test.shape, time.time() - t0))

df_test['score'] = df_score
df_test.sort_values('score', ascending=False, inplace=True)
df_test.to_csv('temp/suchith_test_result.csv')

print("\n\nROC Score: {}".format(roc_auc_score(y_true=df_test['LABEL'], y_score=df_test['score'])))