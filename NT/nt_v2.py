import numpy as np
import random
import pickle

class Network:
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        # biases and weights (Xavier)
        self.biases = [np.zeros((y, 1)) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) * np.sqrt(1/x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    # --- Activations ---
    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))
    
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    # --- Feedforward ---
    def feed_forward(self, A0):
        """A0: input shape (features, batch_size)"""
        cache = {"A0": A0}
        Al = A0
        for l in range(self.num_layers - 2):
            Z = self.weights[l] @ Al + self.biases[l]
            Al = self.sigmoid(Z)
            cache[f"A{l+1}"] = Al
            cache[f"Z{l+1}"] = Z
        # output layer with softmax
        ZL = self.weights[-1] @ Al + self.biases[-1]
        AL = self.softmax(ZL)
        cache[f"A{self.num_layers-1}"] = AL
        cache[f"Z{self.num_layers-1}"] = ZL
        return AL, cache

    # --- Cost (cross-entropy) ---
    def cost(self, Y_hat, Y):
        m = Y.shape[1]
        return -np.sum(Y * np.log(Y_hat + 1e-8)) / m

    # --- Update mini-batch (fully vectorized) ---
    def update_mini_batch(self, X_batch, Y_batch, eta):
        """X_batch: (features, batch_size), Y_batch: (classes, batch_size)"""
        m = X_batch.shape[1]
        AL, cache = self.feed_forward(X_batch)

        # --- Gradients ---
        grads_W = [np.zeros_like(w) for w in self.weights]
        grads_b = [np.zeros_like(b) for b in self.biases]

        # output layer
        dZ = AL - Y_batch
        grads_W[-1] = dZ @ cache[f"A{self.num_layers-2}"].T / m
        grads_b[-1] = np.sum(dZ, axis=1, keepdims=True) / m

        # backward pass for hidden layers
        dA_prev = self.weights[-1].T @ dZ
        for l in range(self.num_layers - 2, 0, -1):
            Z = cache[f"Z{l}"]
            A = cache[f"A{l}"]
            dZ = dA_prev * A * (1 - A)  # sigmoid derivative
            grads_W[l-1] = dZ @ cache[f"A{l-1}"].T / m
            grads_b[l-1] = np.sum(dZ, axis=1, keepdims=True) / m
            if l > 1:
                dA_prev = self.weights[l-1].T @ dZ

        # --- Update weights ---
        self.weights = [w - eta * gw for w, gw in zip(self.weights, grads_W)]
        self.biases = [b - eta * gb for b, gb in zip(self.biases, grads_b)]

    # --- SGD ---
    def sgd(self, X, Y, epochs, mini_batch_size, eta):
        n = X.shape[1]
        for j in range(epochs):
            perm = np.random.permutation(n)
            X_shuffled = X[:, perm]
            Y_shuffled = Y[:, perm]

            for start in range(0, n, mini_batch_size):
                end = start + mini_batch_size
                X_batch = X_shuffled[:, start:end]
                Y_batch = Y_shuffled[:, start:end]
                self.update_mini_batch(X_batch, Y_batch, eta)

            # compute cost after epoch
            AL, _ = self.feed_forward(X)
            c = self.cost(AL, Y)
            print(f"Epoch {j+1}/{epochs}, cost = {c:.4f}")

    # --- Predict ---
    def predict(self, x):
        """x: shape (features, 1)"""
        AL, _ = self.feed_forward(x)
        return np.argmax(AL)

    # --- Save model ---
    def save(self, filename="mnist_model.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(self, f)
