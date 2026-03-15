import cv2
import numpy as np
from NT.nt_v2 import Network
import pickle

# -------------------------
# LOAD TRAINED NETWORK
# -------------------------
with open("mnist_model.pkl", "rb") as f:
    net = pickle.load(f)

# -------------------------
# CAPTURE IMAGE
# -------------------------
cap = cv2.VideoCapture(0)

print("Press SPACE to capture digit")
print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Camera", frame)

    key = cv2.waitKey(1)

    if key == ord(" "):  # space = capture
        img = frame
        break
    elif key == ord("q"):
        cap.release()
        cv2.destroyAllWindows()
        exit()

cap.release()
cv2.destroyAllWindows()

# -------------------------
# PREPROCESS IMAGE
# -------------------------

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# reduce noise
blur = cv2.GaussianBlur(gray, (5,5), 0)

# invert so digit becomes white
thresh = cv2.adaptiveThreshold(
    blur,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    11,
    2
)

# find contours
contours, _ = cv2.findContours(
    thresh,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

if len(contours) == 0:
    print("No digit detected")
    exit()

# largest contour = digit
c = max(contours, key=cv2.contourArea)

x, y, w, h = cv2.boundingRect(c)
digit = thresh[y:y+h, x:x+w]

# -------------------------
# MAKE SQUARE (MNIST STYLE)
# -------------------------

size = max(w, h)
square = np.zeros((size, size), dtype=np.uint8)

x_offset = (size - w) // 2
y_offset = (size - h) // 2

square[y_offset:y_offset+h, x_offset:x_offset+w] = digit

# -------------------------
# RESIZE TO MNIST
# -------------------------

digit = cv2.resize(square, (28,28), interpolation=cv2.INTER_AREA)

digit = digit / 255.0
x_input = digit.reshape(784,1)
# -------------------------
# PREDICT
# -------------------------

prediction = np.argmax(net.feed_forward(x_input))

print("Prediction:", prediction)

cv2.imshow("Digit", digit)
cv2.waitKey(0)
cv2.destroyAllWindows()
