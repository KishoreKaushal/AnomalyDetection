import argparse
import time
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from IsolationForest import IsolationForest
from FeedbackIsolationForest import FeedbackIsolationForest

"""
Input file must be a pickle file containing dataframe 
of shape -> (number of instances, number of features + 1)
The last column must be a boolean data: 1 denotes an anomaly, 0 mean its not.

Output file will contain an array of boolean values, 1 means anomaly, 0 means its not.
"""

def parse_args():
    parser = argparse.ArgumentParser(add_help=True, description='testing feedback guided isolation forest model')
    parser.add_argument('-n', '--ntrees', type=int, default=128, help='number of trees in the forest, default 100')
    parser.add_argument('-s', '--subsamplesize', type=int, default=256, help='sampling rate for each tree, default 256')
    # parser.add_argument('-l', '--loss', type=str, default='linear', help='loss function linear or log-likelihood, default linear')
    parser.add_argument('-lr', '--lrate', type=float, help='learning rate of mirror descent algorithm')

    required_named = parser.add_argument_group('required named arguments')

    required_named.add_argument('-hl', '--hlim', type=int, help='height limit for tree', required=True)
    required_named.add_argument('-i', '--input', type=str, help='path of the input pickle file', required=True)
    # required_named.add_argument('-o', '--output', type=str, help='path for results file', required=True)

    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_pickle(args.input)
    df = shuffle(df)
    df.reset_index(inplace=True, drop=True)

    df_train, df_test = train_test_split(df, test_size=0.01)
    df_train.reset_index(inplace=True, drop=True)
    df_test.reset_index(inplace=True, drop=True)

    print("df_train.shape : ", df_train.shape)

    col = df_train.columns[-1]

    df_x_train, df_y_train = df_train.loc[:, :col], df_train.loc[:, col]
    df_x_test, df_y_test = df_test.loc[:, :col], df_test.loc[:, col]

    test_isolation_forest(df_x_train, df_y_train, df_x_test, df_y_test,
                          args.ntrees, args.subsamplesize, args.hlim)

    test_feedback_isolation_forest(df_x_train, df_y_train, df_x_test, df_y_test,
                                   args.ntrees, args.subsamplesize, args.hlim, args.lrate)


def test_isolation_forest(df_x_train, df_y_train, df_x_test, df_y_test,
                          ntrees, subsample_size, hlim):

    threshold = 0.5

    print("--- Isolation Forest Testing ---")
    print("ntrees: {}, subsample_size: {}, hlim: {}"
          .format(ntrees, subsample_size, hlim))

    iforest = IsolationForest(ntrees, subsample_size, False)

    t0 = time.time()
    iforest.fit(df_x_train)
    print("Training finished in : {}".format(time.time() - t0))
    
    # calculate training accuracy
    # pred = (iforest.anomaly_score(df_x_train, hlim) > threshold).astype(np.uint8)
    # train_acc = (df_y_train == pred).sum()[0] / df_y_train.shape[0]
    # print("train acc : {}".format(hlim, train_acc))

    # calculate test accuracy
    t0 = time.time()
    scores = iforest.anomaly_score(df_x_test, hlim)
    pred = (scores > threshold).astype(np.uint8)
    test_acc = (df_y_test == pred).sum() / df_y_test.shape[0]
    print("test time: {}, test acc : {}".format(time.time()-t0, test_acc))


def test_feedback_isolation_forest(df_x_train, df_y_train, df_x_test, df_y_test,
                                   ntrees, samplesize, hlim, lrate):
    pass

if __name__ == '__main__':
    main()