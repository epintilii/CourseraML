### Week 3 Neural Networks Representation

import os
import scipy.io
import scipy.special
import numpy as np
import matplotlib.pyplot as plt
import random
import math

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

myThetas = [Theta1, Theta2]
n_correct, n_total = 0.0, 0.0
incorect_indices = []

for irow in range(X.shape[0]):
    n_total += 1
    if predictNN(X[irow], myThetas) == y[irow]:
        n_correct +=1
    else:
        incorect_indices.append(irow)

print("Training set accuracy: %s" %(100*(n_correct/n_total)))



# for i in range(5):
#     image_index = random.choice(incorect_indices)
#     image_matrix = X[image_index, 1:].reshape([20, 20]).T
#     plt.imshow(image_matrix, cmap="gray")
#     predicted_val = predictNN(X[image_index], myThetas)
#     predicted_val = 0 if predicted_val == 10 else predicted_val
#     plt.title("Predicted: %d" % predicted_val)
#     plt.show()

###From here starts the Week4 - Neural Networks Learning

input_layer_size = 400
hidden_layer_size = 25
output_layer_size = 10

n_training_samples = X.shape[0]

def flattenParams(myThetas):
    theta_flatten = []
    for theta in myThetas:
        th_fla = theta.flatten()
        theta_flatten.extend(th_fla)
    assert len(theta_flatten) == (input_layer_size+1)*hidden_layer_size +(hidden_layer_size +1) * output_layer_size
    return np.array(theta_flatten).reshape((len(theta_flatten),1))


def reshapeParams(flattened_array):
    theta1 = flattened_array[:(input_layer_size+1)*hidden_layer_size].reshape((hidden_layer_size, input_layer_size+1))
    theta2 = flattened_array[(input_layer_size+1)*hidden_layer_size:].reshape((output_layer_size, hidden_layer_size+1))
    return [theta1, theta2]

def flattenX(myX):
    return np.array(myX.flatten()).reshape((n_training_samples*(input_layer_size+1),1))

def reshapeX(flattenedX):
    return np.array(flattenedX).reshape((n_training_samples, input_layer_size+1))

def computeCost(mythetas_flattened, myX_flattened, myy, mylambda = 0.0):
    mythetas = reshapeParams(mythetas_flattened)
    myX = reshapeX(myX_flattened)
    total_cost = 0.0
    m = n_training_samples
    for irow in range(m):
        myrow = myX[irow]
        myhs = propagateForward(myrow, mythetas)[-1][1]
        tmpy = np.zeros((10,1))
        tmpy[myy[irow]-1] = 1
        cost_myrow = -tmpy.T.dot(np.log(myhs))-(1-tmpy.T).dot(np.log(1-myhs))
        total_cost += cost_myrow
    total_cost = float(total_cost)/m
    total_reg = 0.0
    for mytheta in myThetas:
        total_reg += np.sum(mytheta*mytheta)
    total_reg *= float(mylambda)/(2*m)
    return total_cost + total_reg


def propagateForward(row, Thetas):
    features = row
    zs_as_per_layer = []
    for i in range(len(Thetas)):
        Theta = Thetas[i]
        z = Theta.dot(features).reshape((Theta.shape[0],1))
        a = exponential(z)
        zs_as_per_layer.append((z, a))
        if i == len(Thetas)-1:
            return np.array(zs_as_per_layer)
        a = np.insert(a, 0, 1)
        features = a


def exponential(z):
    a = np.zeros(z.shape)
    for i in range(len(z)):
        sigmoid = 1/(1+math.exp(-z[i]))
        a[i]=sigmoid
    return a


def exponential2(z):
    a = []
    for i in range(len(z)):
        sigmoid = 1/(1+math.exp(-z[i]))
        a.append(sigmoid)
    a = np.array(a).reshape(z.shape)
    return a

print(computeCost(flattenParams(myThetas), flattenX(X), y))

###Backpropagation

def sigmoidGradient(z):
    activation = exponential(z)
    return activation*(1-activation)