import argparse
import time
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from IsolationForest import IsolationForest
from FeedbackIsolationForest import FeedbackIsolationForest
import utils
from copy import deepcopy

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
    parser.add_argument('-lr', '--lrate', type=float, help='learning rate of mirror descent algorithm')
    required_named = parser.add_argument_group('required named arguments')

    required_named.add_argument('-f', '--forest', type=int, help='0 for normal iforest, otherwise feedback iforest', required=True)
    required_named.add_argument('-hl', '--hlim', type=int, help='height limit for tree', required=True)
    required_named.add_argument('-i', '--input', type=str, help='path of the input pickle file', required=True)
    # required_named.add_argument('-o', '--output', type=str, help='path for results file', required=True)

    return parser.parse_args()


def online_update():
    pass

def batch_update():
    pass


def test_feedback_isolation_forest(df, ntrees, subsamplesize, hlim, lrate):
    """
    Parameters
    ----------
    df : pd.DataFrame object
        The last column must contain the feedback from the domain experts.

    ntrees : int
        Number of trees in the forest.

    subsamplesize: int
        SUbsample size for each tree in isolation forest.

    hlim : int
        Height limit.

    lrate : float
        Learning rate of the mirror descent algorithm.
    """
    print("--- Feedback Guided Isolation Forest ---")
    print("Training data shape: ", df.shape)
    print("ntrees: {}, subsample_size: {}, hlim: {}"
          .format(ntrees, subsamplesize, hlim))

    t0 = time.time()
    fif = FeedbackIsolationForest(ntrees, subsamplesize, False, lrate=lrate, df=df)
    print("Initializing base isolation forests time: {}".format(time.time() - t0))

    df_test = df.copy()

    df_test['score'] = 0.0

    # feedback loop
    while(df_test.shape[0] > 1):
        # get score on test dataset
        df_test['score'] = fif.score(df_test[df_test.columns[:-1]], hlim)

        # print max and min anomaly score
        print("min anomaly score: {}\nmax anomaly score: {}"
                .format(df_test['score'].min() , df_test['score'].max()))

        # print top 10 anomalies
        top_10 = df_test.nlargest(10, 'score', keep='first')
        print("Top 10 anomalies : \n", top_10)

        # get max anomaly score and data inst
        idxmax = df_test.idxmax()['score']
        inst = df_test.loc[idxmax]

        #  and drop it from the test dataset
        df_test.drop(labels=idxmax, inplace=True)

        print("Number of dataset left: {}".format(df_test.shape[0]))
        # take feedback and update the weights of the trees
        feedback = int(input("Enter feedback (1, -1) for data instance: \n{} \n=> ".format(inst)))
        fif.update_weights(hlim, feedback, lrate, inst)


def test_isolation_forest(df_x_train, df_y_train, df_x_test, df_y_test,
                          ntrees, subsample_size, hlim, eval_train):
    threshold = 0.5

    print("--- Isolation Forest Testing ---")
    print("Training data shape: ", df_x_train.shape)
    print("ntrees: {}, subsample_size: {}, hlim: {}"
          .format(ntrees, subsample_size, hlim))

    iforest = IsolationForest(ntrees, subsample_size, False)

    t0 = time.time()
    iforest.fit(df_x_train)
    print("Training finished in : {}".format(time.time() - t0))

    # compute sore

    score = iforest.anomaly_score(df_x_train, hlim)

    df = deepcopy(df_x_train)
    df['score'] = score

    df.to_pickle('./results/results_hlim_{}.pickle'.format(hlim))
    # # calculate training accuracy
    # if eval_train:
    #     pred = (iforest.anomaly_score(df_x_train, hlim) > threshold).astype(np.uint8)
    #     train_acc = (df_y_train == pred).sum()[0] / df_y_train.shape[0]
    #     print("train acc : {}".format(hlim, train_acc))
    #
    # if df_y_test.shape[0] <= 20:
    #     # calculate test accuracy
    #     t0 = time.time()
    #     scores = iforest.anomaly_score(df_x_test, hlim)
    #     pred = (scores > threshold).astype(np.uint8)
    #     test_acc = (df_y_test == pred).sum() / df_y_test.shape[0]
    #     print("test time: {}, test acc : {}".format(time.time() - t0, test_acc))


def main():
    args = parse_args()

    df = pd.read_pickle(args.input)
    df = shuffle(df)
    df.reset_index(inplace=True, drop=True)


    # df_train, df_test = train_test_split(df, test_size=args.testsize)
    # df_train.reset_index(inplace=True, drop=True)
    # df_test.reset_index(inplace=True, drop=True)
    #
    # print("df_train.shape : ", df_train.shape)
    #
    # col = df_train.columns[-1]
    #
    # df_x_train, df_y_train = df_train.loc[:, :col], df_train.loc[:, col]
    # df_x_test, df_y_test = df_test.loc[:, :col], df_test.loc[:, col]

    if args.forest == 0:
        pass
        # test_isolation_forest(df_x_train, df_y_train, df_x_test, df_y_test,
        #           args.ntrees, args.subsamplesize, args.hlim, args.evaltrain)
    else:
        test_feedback_isolation_forest(df, args.ntrees,
                    args.subsamplesize, args.hlim, args.lrate)


if __name__ == '__main__':
    main()