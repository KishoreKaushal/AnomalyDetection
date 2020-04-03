import argparse

"""
Input file must be a pickle file containing dataframe 
of shape -> (number of instances, number of features + 1)
The last column must be a boolean data: 1 denotes an anomaly, 0 mean its not.

Output file will contain an array of boolean values, 1 means anomaly, 0 means its not.
"""

def main():
    parser = argparse.ArgumentParser(add_help=True, description='testing feedback guided isolation forest model')
    parser.add_argument('-n', '--ntrees', type=int, default=128, help='number of trees in the forest, default 100')
    parser.add_argument('-s', '--samplesize', type=int, default=256, help='sampling rate for each tree, default 256')
    # parser.add_argument('-l', '--loss', type=str, default='linear', help='loss function linear or log-likelihood, default linear')
    parser.add_argument('-lr', '--lrate', type=float, help='learning rate of mirror descent algorithm')

    required_named = parser.add_argument_group('required named arguments')
    
    required_named.add_argument('-h', '--hlim', type=int, help='height limit for tree', required=True)
    required_named.add_argument('-i', '--input', type=str, help='path of the input file', required=True)
    required_named.add_argument('-o', '--output', type=str, help='path for results file', required=True)
    parser.parse_args()


def test_isolation_forest(df, ntrees, samplesize, hlim):
    pass

def test_feedback_isolation_forest(df, ntrees, samplesize, hlim, lrate):
    pass

if __name__ == '__main__':
    main()