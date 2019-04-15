import features
import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import numpy as np
class DTW(Object):
    def __init__(self, images):
        self.images = images
        self.images_features = features.calculate_features(self.images)
        self.diagonal_margin = 50
    def calculate_cost(self, image_to_compare):
        features_compare = np.array(features.calculate_image_features(image_to_compare))
        score = 0
        for i in range(len(self.images)):
            current_vector = np.array(self.images_features[i])
            distance, path = fastdtw(current_vector, features_compare, dist=euclidean)
            score += distance
        return score / len(self.images)
            