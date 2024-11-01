# neural_networks

Class: NN

The NN class represents a neural network with one input layer, two hidden layers, and one output layer. It uses the sigmoid activation function and trains using gradient descent.


Parameters:
----------------------------------------------------------------------------------------------------------------
input_size (int): Number of input features.
hidden_size (int): Number of neurons in each hidden layer.
output_size (int): Number of output neurons (usually 1 for binary classification).

Methods
-----------------------------------------------------------------------------------------------------------------
compute_loss(y_true, y_pred)
Calculates the mean squared error loss between the true labels and predictions.

Parameters:
y_true (numpy.ndarray): True labels of shape (num_samples, 1).
y_pred (numpy.ndarray): Predicted outputs from the network.
Returns:
float: Mean squared error loss.
-----------------------------------------------------------------------------------------------------------------------------
backward(X, y_true, y_pred)
Performs backpropagation to compute gradients and updates weights and biases.

Parameters:
X (numpy.ndarray): Input data.
y_true (numpy.ndarray): True labels.
y_pred (numpy.ndarray): Predicted outputs.
-----------------------------------------------------------------------------------------------------------------------------
train(X, y, epochs, learning_rate)
Trains the neural network using the provided data.

Parameters:
X (numpy.ndarray): Training data.
y (numpy.ndarray): Training labels.
epochs (int): Number of training iterations.
learning_rate (float): Learning rate for weight updates.
-----------------------------------------------------------------------------------------------------------------------------
predict(X)
Predicts binary class labels for given input data.

Parameters:
X (numpy.ndarray): Input data.
Returns:
numpy.ndarray: Predicted class labels (0 or 1).
-----------------------------------------------------------------------------------------------------------------------------
accuracy(y_true, y_pred)
Calculates the accuracy of the predictions.

Parameters:
y_true (numpy.ndarray): True labels.
y_pred (numpy.ndarray): Predicted labels.
Returns:
float: Accuracy score between 0 and 1.
