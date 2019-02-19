import os
import scipy.io
import scipy.special
import numpy as np

CWD = os.getcwd()
folder_weights = "data\ex3weights.mat"
folder_data = "data\ex3data1.mat"


datafile_weights = os.path.join(CWD, folder_weights)
datafile_data = os.path.join(CWD, folder_data)

weights = scipy.io.loadmat(datafile_weights)
data = scipy.io.loadmat(datafile_data)

X = data["X"]
y = data["y"]

X = np.insert(X,0,1,axis=1)

Theta1, Theta2 = weights['Theta1'], weights['Theta2']
print("Theta1 shape %s . Theta2 shape is %s" %(Theta1.shape, Theta2.shape))

def propagateForward(row, Thetas):
    features = row
    for i in range(len(Thetas)):
        Theta = Thetas[i]
        z = Theta.dot(features)
        a = scipy.special.expit(z)
        if i == len(Thetas)-1:
            return a
        a = np.insert(a, 0, 1)
        features = a

def predictNN(row, Thetas):
    classes = list(range(1, 10)) + [10]
    output = propagateForward(row, Thetas)
    return classes[np.argmax(np.array(output))]


