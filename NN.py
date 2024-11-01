import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
import os



##Implementing the neural network
class NN(object):
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        #Define the wieght matrices and the bias
        self.W1 = np.random.rand(self.input_size, self.hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.rand(self.hidden_size, self.hidden_size)
        self.b2 = np.zeros((1, self.hidden_size))
        self.W3 = np.random.rand(self.hidden_size, self.output_size)
        self.b3 = np.zeros((1, self.output_size))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        self.a3 = self.sigmoid(self.z3)
        self.y_hat = self.a3
        return self.y_hat ### Returning the prediction
      
    def compute_loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred)**2)
    
    def backward(self, X, y_true, y_pred):
        m = y_true.shape[0]  # Number of samples
        print("y_true shape:", y_true.shape)
        print("y_pred shape:", y_pred.shape)
        
        y_true = y_true.reshape(-1, 1)  # Ensure y_true is of shape (m, 1
        # Output Layer.. Compute the derivative of the loss w.r.t y_pred
        dL_dy_pred = 2 * (y_pred - y_true) / m  # (m, o)
        
        # Compute dz3
        dz3 = dL_dy_pred * self.sigmoid_derivative(y_pred)  # (m, o)
        print("dz3 shape")
        # Compute gradients for W3 and b3
        dW3 = np.dot(self.a2.T, dz3)  # (hidden_size2, o)
        db3 = np.sum(dz3, axis=0, keepdims=True)  # (1, o)
        
        # Hidden Layer Propagate the error back to Hidden Layer 2
        dz2 = np.dot(dz3, self.W3.T) * self.sigmoid_derivative(self.a2)  # (m, h2)
        
        # Compute gradients for W2 and b2
        dW2 = np.dot(self.a1.T, dz2)  # (hidden_size1, hidden_size2)
        db2 = np.sum(dz2, axis=0, keepdims=True)  # (1, h2)
        
   
        # Hidden Layer 1.. Propagate the error back to Hidden Layer 1
        dz1 = np.dot(dz2, self.W2.T) * self.sigmoid_derivative(self.a1)  # (m, h1)
        
        # Compute gradients for W1 and b1
        dW1 = np.dot(X.T, dz1)  # (input_size, h1)
        db1 = np.sum(dz1, axis=0, keepdims=True)  # (1, h1)
        
      
        # Update Weights and Biases
       
        self.W3 -= self.learning_rate * dW3
        self.b3 -= self.learning_rate * db3
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
      
    def train(self, X, y, epochs, learning_rate):
        self.learning_rate = learning_rate
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = self.compute_loss(y, y_pred)
            self.backward(X, y, y_pred)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, X):
        y_pred = self.forward(X)
        return (y_pred > 0.5).astype(int)
    
    def accuracy(self, y_true, y_pred):
        return np.mean(np.round(y_pred) == y_true)



    

      
      

    

