import cv2

faces = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

#creates video capture with set resolution and brightness
cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
cap.set(10,100)

#continues reading images to feed video capture.
while True:
    success, img = cap.read()

    #converts colored images to grayscale
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #detects faces
    detections = faces.detectMultiScale(imgGray,scaleFactor=1.1,minNeighbors=6)

    #places rectangle around faces
    for (x,y,w,h) in detections:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

    #shows every image in the form of a "video"
    cv2.imshow("Video", img)
    
    # quits program on q press
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break
