import cv2
import numpy as np


class DataSet:

    def __init__(self, file):
        with open(file, "r") as f:
            self.data = f.read().split("\n")
        self._num_examples = len(self.data)
        self.X = np.ones([self._num_examples, 224, 224, 3])
        self.y = np.ones([self._num_examples, 5])
        self._indicator = 0

        for i, data_i in enumerate(self.data):
            annotation, y = data_i.split(" ")
            image = self._get_image(annotation)
            self.X[i] = image
            self.y[i] = np.eye(5)[int(y)]

    def _get_image(self, annotation):
        image = cv2.imread(annotation)
        image = np.array(cv2.resize(image, (224, 224))).astype(np.float)
        _R_MEAN = 123.68
        _G_MEAN = 116.78
        _B_MEAN = 103.94
        image[:, :, 0] -= _R_MEAN
        image[:, :, 1] -= _G_MEAN
        image[:, :, 2] -= _B_MEAN
        return image
