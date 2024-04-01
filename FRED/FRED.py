import cv2
from deepface import DeepFace

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

video = cv2.VideoCapture(0)

if not video.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = video.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray, 1.1, 3)

    for x, y, w, h in face:
        image = cv2.rectangle(frame, (x, y), (x + w, y + h), (89, 2, 236), 1)
        try:
            analyze = DeepFace.analyze(frame, actions=['emotion'])
            cv2.putText(image, analyze[0]['dominant_emotion'], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (224, 77, 176), 2)
            print(analyze[0]['dominant_emotion'])
        except:
            print('no face')

    cv2.imshow('FRED v0.2', frame)
    key = cv2.waitKey(33)
    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()