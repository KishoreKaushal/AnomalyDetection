import numpy as np

class Histogram():

    def __init__(self, arr, count, max_buckets, eps):
        self.num = len(arr)
        self.max_buckets = max_buckets
        self.arr = arr
        self.eps = eps
        self.err, self.b_values = ahist_s(arr, count, max_buckets, eps)

    def get_buckets(self, num_buckets):
        buckets = []
        end = self.num - 1
        k = num_buckets - 1
        while end >= 0:
            start = int(self.b_values[k][end][0])
            if start <= end:
                buckets.append(start)
            end = start - 1
            k -= 1
        return np.flip(buckets, axis=0)

    def best_split(self):
        if self.err[0] == 0:
            return 0, 0, []
        err_red = [(self.err[0] - self.err[i]) for i in range(1, self.max_buckets)]
        var_red = np.max(err_red) / self.err[0]
        if var_red < 0:
            print("error: var_red is", var_red)
            var_red = 0
        opt = np.argmax(err_red) + 2
        buckets = self.get_buckets(opt)
        return opt, var_red, buckets[1:]



def ahist_s(arr, count, max_buckets, eps):

    approx_err = np.zeros(max_buckets) - 1

    # error of best histogram with k buckets
    err_of_best_hist = np.zeros(max_buckets)

    b_values = [dict() for _ in range(max_buckets)]

    accumulated_sum = 0
    accumulated_sqr_sum = 0
    seen_points = 0

    for j in range(len(arr)):
        accumulated_sum += arr[j] * count[j]
        accumulated_sqr_sum += (arr[j]**2) * count[j]
        seen_points += count[j]


        # for one bucket minimum error is with mean of data points
        # refer Guha Et. Al 2006 -- equation 2
        # err_of_best_hist[0] -- means error of best histogram with only one bucket

        err_of_best_hist[0] = accumulated_sqr_sum - (accumulated_sum**2)/seen_points

        if err_of_best_hist[0] > (1 + eps) * approx_err[0]:
            approx_err[0] = err_of_best_hist[0]
        else:
            del b_values[0][j-1]

        b_values[0][j] = tuple([ 0,
                                 err_of_best_hist[0],
                                 accumulated_sum,
                                 accumulated_sqr_sum,
                                 seen_points ])

        for k in range(1, max_buckets):
            err_of_best_hist[k] = err_of_best_hist[k-1]
            a_val = j + 1

            for b_val in b_values[k-1].keys():
                if b_val < j:
                    _, b_err, b_sum, b_sqr, b_pts = b_values[k-1][b_val]
                    tmp_err = b_err + accumulated_sqr_sum - b_sqr \
                              - (accumulated_sum - b_sum)**2 / (seen_points - b_pts)

                    if tmp_err < err_of_best_hist[k]:
                        err_of_best_hist[k] = tmp_err
                        a_val = b_val + 1

            b_values[k][j] = tuple([ a_val,
                                     err_of_best_hist[k],
                                     accumulated_sum,
                                     accumulated_sqr_sum,
                                     seen_points ])

            if err_of_best_hist[k] > (1 + eps) * approx_err[k]:
                approx_err[k] = err_of_best_hist[k]
            else:
                del b_values[k][j-1]

    return err_of_best_hist, b_values



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



