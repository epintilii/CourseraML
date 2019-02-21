import os
import scipy.io
import scipy.special
import numpy as np
import matplotlib.pyplot as plt
import random
import itertools

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



for i in range(5):
    image_index = random.choice(incorect_indices)
    image_matrix = X[image_index, 1:].reshape([20, 20]).T
    plt.imshow(image_matrix, cmap="gray")
    predicted_val = predictNN(X[image_index], myThetas)
    predicted_val = 0 if predicted_val == 10 else predicted_val
    plt.title("Predicted: %d" % predicted_val)
    plt.show()


input_layer_size = 400
hidden_layer_size = 25
output_layer_size = 10

n_training_samples = X.shape[0]

def flattenParams(thetas_list):
    flattened_list = [mytheta.flatten() for mytheta in thetas_list]
    combined = list(itertools.chain.from_iterable(flattened_list))
    assert len(combined) == (input_layer_size+1)*hidden_layer_size + (hidden_layer_size +1) * output_layer_size
    return np.array(combined).reshape((len(combined),1))

a = Theta1.flatten()
b = list(itertools.chain.from_iterable(a))
print(b)