import pandas as pd

train_data_path = "~/dataset/train_24072020.pkl"
test_data_path = "~/dataset/test_24072020.pkl"

df_train = pd.read_pickle(train_data_path)
df_test = pd.read_pickle(test_data_path)

# preprocessing
day_dict = {
    'SUN' : 1,
    'MON' : 2,
    'TUE' : 3,
    'WED' : 4,
    'THU' : 5,
    'FRI' : 6,
    'SAT' : 7
}

def pre_process(df, inplace=True):
    if not inplace:
        df = df.copy()

    df['APIKEY'] = df['APIKEY'].apply(lambda x: x.replace("APIKEY", "")).astype(int)
    df['TIMEBIN'] = df['TIMEBIN'].apply(lambda x: x.replace("BIN", "")).astype(int)
    df['LABEL'] = df['LABEL'].apply(lambda x: 0 if x == 'NOT ANAMOLY' else 1)
    df['DAY'] = df['DAY'].apply(lambda x : day_dict[x]).astype(int)
    df.drop(columns=['ANAMOLYDISTNUM'], inplace=True)
    df = df.dropna()
    return df

pre_process(df_train)
pre_process(df_test)

df_train.to_pickle("~/dataset/preproc_train_24072020.pkl")
df_test.to_pickle("~/dataset/preproc_test_24072020.pkl")

