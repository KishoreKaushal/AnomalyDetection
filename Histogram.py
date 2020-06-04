import numpy as np

class Histogram():

    def __init__(self, val, count, max_buckets, eps):
        self.num = len(val)
        self.max_buckets = max_buckets
        self.val = val
        self.eps = eps

    def get_buckets(self, num_buckets):
        pass





#
#
# def ahist_s(arr, max_buckets, eps):
#     apxerr = np.zeros(len(arr)+1, max_buckets+1)*np.inf
#     Q = [[[0,0,np.inf, ]] for _ in range(max_buckets+1)]
#     sqerr = np.zeros(max_buckets + 1, len(arr) + 1)
#     sum = sqsum = 0.0
#
#     for j in range(1,len(arr) + 1):
#         sum += arr[j]
#         sqsum += arr[j]**2
#         apxerr[j, 1] = sqerr[1, j] = sqsum - sum**2 / j    # equation 2  of guha et al. 2006
#
#         for k in range(2, max_buckets+1):
#             apxerr[j,k] = np.inf
#
#             for ele in Q[k-1]:
#                 i = ele[1]
#                 apxerr[j,k] = min(apxerr[j,k], apxerr[i,k-1] + sqerr[i+1, j])
#
#             # a_l is the start index of the last interval in Q[k]
#             # b_l is the end  index of the last interval in Q[k]
#
#             if len(Q[k]) != 0:
#                 a_l = Q[k][-1][0]
#             else:
#                 a_l = 0
#
#             if (k <= max_buckets - 1 and apxerr[j,k] > (1+eps) * apxerr[a_l, k]):
#                 # now we have apxerr[j,k] , sm = sum[j], sqsm = sqsum[j]
#                 a_l_1 = b_l_1 = j
#                 Q[k].append([a_l_1, b_l_1, ])
#             else:
#                 b_l = j



