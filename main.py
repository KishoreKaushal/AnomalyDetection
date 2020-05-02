import argparse
import time
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from IsolationForest import IsolationForest
from FeedbackIsolationForest import FeedbackIsolationForest
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
    parser.add_argument('-tp', '--top', type=int, default=10, help='print top anomalies in each loop of iff')
    parser.add_argument('-td', '--testdata', type=str, default='', help='path of the test pickle file')
    parser.add_argument('-ec', '--excludecolidx', type=str, default='', help='columns to exclude for training')
    required_named = parser.add_argument_group('required named arguments')

    required_named.add_argument('-f', '--forest', type=int, help='0 for normal iforest, otherwise feedback iforest', required=True)
    required_named.add_argument('-hl', '--hlim', type=int, help='height limit for tree', required=True)
    required_named.add_argument('-i', '--input', type=str, help='path of the input pickle file', required=True)

    return parser.parse_args()


def online_update():
    pass

def batch_update():
    pass


def test_feedback_isolation_forest(df_train, df_test, ntrees, subsamplesize, hlim, lrate, tp):
    """
    Parameters
    ----------
    df_train : pd.DataFrame object
        Training dataset.

    df_test : pd.DataFrame object
        Testing dataset. The last column must
        contain the feedback from the domain experts.

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
    print("Training data shape: ", df_train.shape)
    print("ntrees: {}, subsample_size: {}, hlim: {}"
          .format(ntrees, subsamplesize, hlim))

    t0 = time.time()
    fif = FeedbackIsolationForest(ntrees, subsamplesize, False, lrate=lrate, df=df_train)
    print("Initializing base isolation forests time: {}".format(time.time() - t0))

    df_test = df_test.copy()

    df_test['score'] = 0.0

    # feedback loop
    while(df_test.shape[0] > 1):
        # get score on test dataset
        df_test['score'] = fif.score(df_test[df_test.columns[:-1]], hlim)

        # print max and min anomaly score
        print("min anomaly score: {}\nmax anomaly score: {}"
                .format(df_test['score'].min() , df_test['score'].max()))

        # print top 10 anomalies
        top_10 = df_test.nlargest(tp, 'score', keep='first')
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


def main():
    args = parse_args()

    df = pd.read_pickle(args.input)

    # generate the list of exluded columns for training
    exclude_col_for_train = []
    for x in args.excludecolidx.split(','):
        if x.strip() != '':
            exclude_col_for_train.append(df.columns[int(x.strip())])

    df_train = df.drop(columns=exclude_col_for_train)

    if args.testdata == '':
        df_test = df_train
    else:
        df_test = pd.read_pickle(args.testdata)

    if args.forest == 0:
        pass
    else:
        test_feedback_isolation_forest(df_train, df_test, args.ntrees,
                    args.subsamplesize, args.hlim, args.lrate, args.top)


if __name__ == '__main__':
    main()