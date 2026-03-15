import numpy as np

# --- LOAD CSV ---
data = np.loadtxt("mnist_train.csv", delimiter=",")  # shape: (m, 785)
Y_raw = data[:, 0].astype(int)      # first column = labels
X_raw = data[:, 1:]                 # remaining 784 columns = pixels

# --- NORMALIZE INPUT ---
X = (X_raw / 255.0).T               # shape: 784 x m
m = X.shape[1]

# --- ONE-HOT ENCODE LABELS ---
num_classes = 10
Y = np.zeros((num_classes, m))
for i, label in enumerate(Y_raw):
    Y[label, i] = 1

# --- NETWORK DIMENSIONS ---
n = [784, 128, 64, 32, 10]  # 3 hidden layers
L = len(n) - 1

# --- INITIALIZE WEIGHTS AND BIASES (Xavier) ---
W = [None] + [np.random.randn(n[l+1], n[l]) * np.sqrt(1/n[l]) for l in range(L)]
b = [None] + [np.zeros((n[l+1], 1)) for l in range(L)]

# --- ACTIVATIONS ---
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

# --- COST ---
def cost(Y_hat, Y):
    m = Y.shape[1]
    return -np.sum(Y * np.log(Y_hat + 1e-8)) / m

# --- FEEDFORWARD ---
def feed_forward(A0):
    cache = {"A0": A0}
    Al = A0
    for l in range(1, L):
        Z = W[l] @ Al + b[l]
        Al = sigmoid(Z)
        cache[f"A{l}"] = Al
    ZL = W[L] @ Al + b[L]
    AL = softmax(ZL)
    cache[f"A{L}"] = AL
    return AL, cache

# --- TRAINING WITH MINI-BATCHES ---
def train(batch_size=128, epochs=5, alpha=0.1):
    global W, b
    num_samples = X.shape[1]

    for e in range(epochs):
        perm = np.random.permutation(num_samples)  # shuffle dataset
        X_shuffled = X[:, perm]
        Y_shuffled = Y[:, perm]

        for start in range(0, num_samples, batch_size):
            end = start + batch_size
            X_batch = X_shuffled[:, start:end]
            Y_batch = Y_shuffled[:, start:end]
            m_batch = X_batch.shape[1]

            # forward pass
            Y_hat, cache = feed_forward(X_batch)

            # compute gradients (softmax + cross-entropy)
            dZ = Y_hat - Y_batch
            grads_W = [None]*(L+1)
            grads_b = [None]*(L+1)

            grads_W[L] = dZ @ cache[f"A{L-1}"].T / m_batch
            grads_b[L] = np.sum(dZ, axis=1, keepdims=True) / m_batch

            dA_prev = W[L].T @ dZ
            for l in range(L-1, 0, -1):
                dZ = dA_prev * cache[f"A{l}"] * (1 - cache[f"A{l}"])
                grads_W[l] = dZ @ cache[f"A{l-1}"].T / m_batch
                grads_b[l] = np.sum(dZ, axis=1, keepdims=True) / m_batch
                if l > 1:
                    dA_prev = W[l].T @ dZ

            # update weights
            for l in range(1, L+1):
                W[l] -= alpha * grads_W[l]
                b[l] -= alpha * grads_b[l]

        # compute full-batch cost after epoch
        Y_hat_full, _ = feed_forward(X)
        c = cost(Y_hat_full, Y)
        print(f"Epoch {e+1}/{epochs}, cost = {c:.4f}")

# --- TRAIN ---
train(batch_size=128, epochs=50, alpha=0.1)

# --- PREDICT EXAMPLE ---
def predict(x):
    x = (x / 255.0).reshape(-1,1)  # ensure shape 784x1
    Y_hat, _ = feed_forward(x)
    return np.argmax(Y_hat)

# Example: predict first training sample
while True:
    asd = int(input(":: "))
    if (asd >= 60000): asd = 59999
    if (asd < 0): asd = 0
    
    print("Predicted:", predict(X_raw[asd]), "Actual:", Y_raw[asd])
