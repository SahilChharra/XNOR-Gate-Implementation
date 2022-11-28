# DAA-Assignment-2

### Name:- Sahil.R.Chharra
### Roll no :- A-61
### Subject :- Design and Analysis of Alogorithms
### Problem Statement :- Given an application and its implementation to demonstrate the implementation of XNOR gate [Digital Ckts]

### Logic/Thoery:-

Implementation of Artificial Neural Network for XNOR Logic Gate with 2-bit Binary Input
We implemente the logic of XNOR Logic gate to check for the snapses of brain cells. It's with the forward propagation function, the output is predicted. Then through backpropagation, the weight and bias to the nodes are updated to minimizing the error in prediction to attain the convergence of cost function in determining the final output.

### Code :-
```
import numpy as np
from matplotlib import pyplot as plt

def forwardPropagation(X, Y, parameters):
	m = X.shape[1]
	W1 = parameters["W1"]
	W2 = parameters["W2"]
	b1 = parameters["b1"]
	b2 = parameters["b2"]
	Z1 = np.dot(W1, X) + b1
	A1 = sigmoid(Z1)
	Z2 = np.dot(W2, A1) + b2
	A2 = sigmoid(Z2)
	cache = (Z1, A1, W1, b1, Z2, A2, W2, b2)
	logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), (1 - Y))
	cost = -np.sum(logprobs) / m
	return cost, cache, A2

def backwardPropagation(X, Y, cache):
	m = X.shape[1]
	(Z1, A1, W1, b1, Z2, A2, W2, b2) = cache
	dZ2 = A2 - Y
	dW2 = np.dot(dZ2, A1.T) / m
	db2 = np.sum(dZ2, axis = 1, keepdims = True)
	dA1 = np.dot(W2.T, dZ2)
	dZ1 = np.multiply(dA1, A1 * (1- A1))
	dW1 = np.dot(dZ1, X.T) / m
	db1 = np.sum(dZ1, axis = 1, keepdims = True) / m
	
	gradients = {"dZ2": dZ2, "dW2": dW2, "db2": db2,
				"dZ1": dZ1, "dW1": dW1, "db1": db1}
	return gradients
	
def sigmoid(z):
	return 1 / (1 + np.exp(-z))

X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]]) 
Y = np.array([[1, 0, 0, 1]]) 

neuronsInHiddenLayers = 2 
inputFeatures = X.shape[0] 
outputFeatures = Y.shape[0]
parameters = initializeParameters(inputFeatures, neuronsInHiddenLayers, outputFeatures)
epoch = 100000
learningRate = 0.01
losses = np.zeros((epoch, 1))

for i in range(epoch):
	losses[i, 0], cache, A2 = forwardPropagation(X, Y, parameters)
	gradients = backwardPropagation(X, Y, cache)
	parameters = updateParameters(parameters, gradients, learningRate)

plt.figure()
plt.plot(losses)
plt.xlabel("EPOCHS")
plt.ylabel("Loss value")
plt.show()

X = np.array([[0, 0, 0, 0], [0, 0, 0, 1]])
cost, _, A2 = forwardPropagation(X, Y, parameters)
prediction = (A2 > 0.5) * 1.0
print(prediction)
```

### Output Screenshots:-

<p><img src="https://lh5.googleusercontent.com/vFT5L5xZ5optACNBVtpRdGEGMGU6_6dfWY1vila7XLVyJHYjLFg5zRfEZlBbubNyO2s=w1200" height=" 100%"></p>
<p><img src="https://lh5.googleusercontent.com/fSdMVQty7zqz7J6b8-Q3AvgRx04D9LBrm1zeuAdGrcdtU--CWmBgSPhlOFTm-3YUKZU=w2400" height=" 100%"></p>
<p><img src="https://lh4.googleusercontent.com/b3Esj8EJbX0dTDm9WPKFLnccnec2PB19tIXXPxSvzOrr1ethhgaYmRZOF9GmgmbxfIw=w2400" height=" 100%"></p>
