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


# for i in index:
#     image = X[i, 1:]
#     image = np.reshape(image, [20, 20])
#     plt.imshow(image.T, cmap="gray")
#     plt.show()


def getDatumImg(row):
    width, height = 20, 20
    square = row[1:].reshape(width, height)
    return square.T

def displayData(indices_to_display = None):
    width, height = 20, 20
    nrows, ncols = 10, 10
    if not indices_to_display:
        indices_to_display = random.sample(range(X.shape[0]), nrows*ncols)

    big_picture = np.zeros((height*nrows, width*ncols))

    irow, icol = 0, 0
    for idx in indices_to_display:
        if icol == irow:
            irow += 1
            icol = 0
        img = getDatumImg(X[idx])
        big_picture[irow*height:irow*height+img.shape[0], icol*width:icol*width+img.shape[1]]
        icol +=1
    fig = plt.figure(figsize=(6,6))
    imgg = scipy.misc.toimage(big_picture)
    plt.imshow(imgg, cmap=cm.Greys_r)
    plt.show()

displayData()

def h(mytheta, myX):
    return expit(np.dot(myX, mytheta))

def compute_cost(mytheta, myX, myy, mylambda = 0.):
    m = myX.shape[0]
    myh = h(mytheta, myX)
    term1 = np.log(myh).dot(-myy.T)
    term2 = np.log(1.0-myh).dot(1-myy.T)
    left_hand = (term1-term2)/m
    right_hand = mytheta.T.dot(mytheta) * mylambda / (2*m)
    return left_hand + right_hand

