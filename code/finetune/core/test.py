from core.dataset_distillation import Distillation_DataSet


train_data = Distillation_DataSet("../data/train_data.txt", "../npy/y_prob.npy")
a, b, c = train_data.next_batch(2)
print(a.shape, b.shape, c.shape)
print(a)
print(b)
print(c)
# import numpy as np
#
# y_prob = np.load("../npy/y_prob.npy")
# print(y_prob.shape)
# print(y_prob)
