import os
import cv2
import numpy as np
from keras_preprocessing import image
import warnings
from keras.models import load_model
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

model = load_model("emodet.h5")

face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

# Define the list of emotions
emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']


while True:
    ret, test_img = cap.read()
    if not ret:
        continue
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
        roi_gray = gray_img[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=-1)
        img_pixels = img_pixels / 255.0

        predictions = model.predict(np.expand_dims(img_pixels, axis=0))

        max_index = np.argmax(predictions[0])
        predicted_emotion = emotions[max_index]

        # Get percentage confidence
        percentage_confidence = predictions[0][max_index] * 100

        cv2.putText(test_img, f'{predicted_emotion} ({percentage_confidence:.2f}%)', (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Facial emotion ', resized_img)

    if cv2.waitKey(10) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()