import numpy as np
from NT.nt_v2 import Network

def vectorized_result(j):
    e = np.zeros((10,1))
    e[j] = 1.0
    return e

# --- LOAD DATA ---
train_data = np.loadtxt("mnist/mnist_train.csv", delimiter=",")
test_data = np.loadtxt("mnist/mnist_test.csv", delimiter=",")

Y_train = train_data[:,0].astype(int)
X_train = train_data[:,1:] / 255.0

Y_test = test_data[:,0].astype(int)
X_test = test_data[:,1:] / 255.0

# --- FORMAT DATA ---
training_data = [
    (x.reshape(784,1), vectorized_result(y))
    for x,y in zip(X_train, Y_train)
]

test_data = [
    (x.reshape(784,1), y)
    for x,y in zip(X_test, Y_test)
]

# --- CREATE NETWORK ---
net = Network([784, 30, 10])

# --- TRAIN ---
net.sgd(
    training_data,
    epochs=30,
    mini_batch_size=10,
    eta=3.0,
    test_data=test_data
)
net.save()

while True:
    user_input = input(f"Enter image index (0-{len(X_test)-1}) or 'q' to quit: ")

    # quit option
    if user_input.lower() == "q":
        break

    # check if number
    if not user_input.isdigit():
        print("Invalid input. Please enter a number.")
        continue

    idx = int(user_input)

    # bounds check
    if idx < 0 or idx >= len(X_test):
        print(f"Index out of bounds. Enter a value between 0 and {len(X_test)-1}.")
        continue

    # run prediction
    x = X_test[idx].reshape(784,1)
    prediction = np.argmax(net.feed_forward(x))

    print("Prediction:", prediction)
    print("Label:", Y_test[idx])
    print()
