import cv2
import numpy as np
from keras_preprocessing import image
from keras.models import load_model
import matplotlib.pyplot as plt

model = load_model("emodet.h5")
face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
graph_saved = False

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

        predictions = model.predict(np.expand_dims(img_pixels, axis=0))[0]

        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        
        if not graph_saved:
            graph_img = np.zeros((400, 800, 3), dtype=np.uint8)
            plt.figure(figsize=(8, 4))
            plt.bar(emotions, predictions)
            plt.title('Emotion Probabilities')
            plt.xlabel('Emotion')
            plt.ylabel('Probability')
            plt.savefig('graph.png')
            graph_saved = True

        # Annotate emotions on the camera feed
        for i, emotion in enumerate(emotions):
            text = f'{emotion}: {predictions[i]:.2f}'
            cv2.putText(test_img, text, (10, 30 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Camera Feed with Graph', test_img)

    if cv2.waitKey(10) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Now save the graph after the camera feed loop exits
if graph_saved:
    graph_img = cv2.imread('graph.png')
    cv2.imwrite('final_graph.png', graph_img)

