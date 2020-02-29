import numpy as np
import cv2


_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

image = cv2.imread("./3.png")
image = np.array(cv2.resize(image, (224, 224))).reshape([1, 224, 224, 3]).astype(np.float)
image[:, :, 0] -= _R_MEAN
image[:, :, 1] -= _G_MEAN
image[:, :, 2] -= _B_MEAN

np.save("./npy/1.npy", image)
