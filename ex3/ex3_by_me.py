import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io #Used to load the OCTAVE *.mat files
import scipy.misc #Used to show matrix as an image
import matplotlib.cm as cm #Used to display images in a specific colormap
import random #To pick random images to display
from scipy.special import expit #Vectorized sigmoid function

CWD = os.getcwd()

folder = "data/ex3data1.mat"

datafile = os.path.join(CWD, folder)


# scipy.io.loadmat returns a dictionary
mat = scipy.io.loadmat(datafile)

# X, y will become numpy arrays
X, y = mat["X"], mat["y"]

# Insert a column of 1's to X as usual
X = np.insert(X, 0, 1, axis=1)

print("y shape: %s. Unique elments in y: %s" %(y.shape, np.unique(y)))
print("X shape: %s. Number of images are: %s" %(X.shape, X.shape[0]))

#X is 5000 images. Each image is a row. Each image has 400 pixels unrolled (20x20)
#y is a classification for each image. 1-10, where "10" is the handwritten "0"

index = np.random.randint(5000, size=10)


for i in index:
    image = X[i, 1:]
    image = np.reshape(image, [20, 20])
    plt.imshow(image.T, cmap="gray")
    plt.show()


def getDatumImg(row):
    width, height = 20, 20
    square = row[1:].reshape(width, height)
    return square.T


