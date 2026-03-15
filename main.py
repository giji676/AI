import time
import numpy as np
from NT.nt_v2 import Network

# --- ONE-HOT ENCODING ---
def vectorized_result(y, num_classes=10):
    e = np.zeros((num_classes, 1))
    e[y] = 1.0
    return e

# --- LOAD DATA ---
train_data = np.loadtxt("mnist/mnist_train.csv", delimiter=",")
test_data = np.loadtxt("mnist/mnist_test.csv", delimiter=",")

# Split labels and pixels
Y_train_raw = train_data[:, 0].astype(int)
X_train = (train_data[:, 1:] / 255.0).T  # shape: (784, num_train)
Y_train = np.hstack([vectorized_result(y) for y in Y_train_raw])  # shape: (10, num_train)

Y_test_raw = test_data[:, 0].astype(int)
X_test = (test_data[:, 1:] / 255.0).T  # shape: (784, num_test)

# --- CREATE NETWORK ---
net = Network([784, 30, 10])

start_time = time.time()
print("Training started...")

# --- TRAIN ---
net.sgd(
    X=X_train,
    Y=Y_train,
    epochs=30,
    mini_batch_size=10,
    eta=3.0
)

end_time = time.time()
duration = end_time - start_time
print(f"Training finished! Total time: {duration:.2f} seconds")

# Save model
net.save()

# --- INTERACTIVE PREDICTION ---
while True:
    user_input = input(f"Enter image index (0-{X_test.shape[1]-1}) or 'q' to quit: ")

    if user_input.lower() == "q":
        break
    if not user_input.isdigit():
        print("Invalid input. Enter a number.")
        continue

    idx = int(user_input)
    if idx < 0 or idx >= X_test.shape[1]:
        print(f"Index out of bounds. Enter 0-{X_test.shape[1]-1}")
        continue

    # reshape single test image for Network
    x = X_test[:, idx].reshape(-1, 1)
    prediction = net.predict(x)
    label = Y_test_raw[idx]

    print("Prediction:", prediction)
    print("Label:", label)
    print()
