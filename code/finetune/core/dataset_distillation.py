import cv2
import numpy as np


class Distillation_DataSet:

    def __init__(self, file, y_prob_npy_file):
        with open(file, "r") as f:
            self.data = f.read().split("\n")
        self._num_examples = len(self.data)
        self.y_prob = np.load(y_prob_npy_file)
        self.X = np.ones([self._num_examples, 56, 56, 3])
        self.y = np.ones([self._num_examples, 5])
        self._indicator = 0

        for i, data_i in enumerate(self.data):
            annotation, y = data_i.split(" ")
            image = self._get_image(annotation)
            self.X[i] = image
            self.y[i] = np.eye(5)[int(y)]
        self._shuffle_data()

    def _shuffle_data(self):
        p = np.random.permutation(self._num_examples)
        self.X = self.X[p]
        self.y = self.y[p]
        self.y_prob = self.y_prob[p]

    def _get_image(self, annotation):
        image = cv2.imread(annotation)
        image = np.array(cv2.resize(image, (56, 56))).astype(np.float)
        _R_MEAN = 123.68
        _G_MEAN = 116.78
        _B_MEAN = 103.94
        image[:, :, 0] -= _R_MEAN
        image[:, :, 1] -= _G_MEAN
        image[:, :, 2] -= _B_MEAN
        return image

    def next_batch(self, batch_size):
        end_indicator = self._indicator + batch_size
        if end_indicator >= self._num_examples:
            self._shuffle_data()
            self._indicator = 0
            end_indicator = batch_size
        if end_indicator > self._num_examples:
            raise Exception("batch size is larger than all examples")
        batch_X = self.X[self._indicator: end_indicator]
        batch_y = self.y[self._indicator: end_indicator]
        batch_prob = self.y_prob[self._indicator: end_indicator]
        self._indicator = end_indicator
        return batch_X, batch_y, batch_prob
