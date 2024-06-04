import time

import mss
import numpy as np
import cv2
import joblib

# Załaduj przetrenowany model KNN
knn_model = joblib.load('knn_digit_recognition_model.pkl')

mss_instance = mss.mss()


def take_ss():
    screenshot = mss_instance.grab({
        "left": 1055,
        "top": 900,
        "width": 200,
        "height": 40
    })

    img = np.array(screenshot)
    return img


def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digits = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        digit = thresh[y:y + h, x:x + w]
        digit = cv2.resize(digit, (28, 28))
        digit = digit.flatten() / 255.0
        digits.append((digit, (x, y, w, h)))
    return digits


def predict_number(model, digits):
    predictions = []
    for digit, bbox in digits:
        digit = digit.reshape(1, -1)
        predicted_number = model.predict(digit)[0]
        predictions.append((predicted_number, bbox))
    return predictions

time.sleep(4)
while True:
    img = take_ss()
    digits = preprocess_image(img)
    predictions = predict_number(knn_model, digits)

    for predicted_number, (x, y, w, h) in predictions:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, str(predicted_number), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("img", img)
    print(predictions)
    if cv2.waitKey(5) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        break
