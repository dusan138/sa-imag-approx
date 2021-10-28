import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

from clustering.kmeans import perform_kmeans

rootdir = os.path.dirname(os.path.realpath(__file__))
imagdir = os.path.normpath(os.path.join(rootdir, '..', 'images'))
print(imagdir)

image_name = 'test_image.jpg'

img = cv2.imread(os.path.join(imagdir, image_name))

vectorized = np.reshape(img, (-1, 3))

perform_kmeans(vectorized)

