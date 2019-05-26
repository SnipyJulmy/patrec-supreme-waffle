import scipy.spatial.distance as dist2
import numpy as np
from collections import defaultdict
from fastdtw import fastdtw
class DTW():
    def __init__(self, features):
        self.features = features
        self.xx = np.asanyarray(self.features, dtype='float')

    def __dtw(self, x,y,dist):
        len_x, len_y = len(x), len(y)
        window = [(i, j) for i in range(len_x) for j in range(len_y)]
        window = ((i + 1, j + 1) for i, j in window)
        D = defaultdict(lambda: (float('inf'),))
        D[0, 0] = (0, 0, 0)
        for i, j in window:
            dt = dist(x[i-1], y[j-1])
            D[i, j] = min((D[i-1, j][0]+dt, i-1, j), (D[i, j-1][0]+dt, i, j-1),
                        (D[i-1, j-1][0]+dt, i-1, j-1), key=lambda a: a[0])
        # path = []
        # i, j = len_x, len_y
        # while not (i == j == 0):
        #     path.append((i-1, j-1))
        #     i, j = D[i, j][1], D[i, j][2]
        # path.reverse()
        return (D[len_x, len_y][0])

    def calculate_cost(self, features):
        features_compare = np.array(features)
        yy = np.asanyarray(features_compare, dtype='float')
        #distance, path = fastdtw(self.xx, yy, dist=dist2.euclidean)
        distance = self.__dtw(self.xx, yy, dist2.euclidean)
        return distance
            