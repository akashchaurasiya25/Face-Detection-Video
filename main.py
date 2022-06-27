import cv2
from random import randrange as r
#dataset load
trainedData = cv2.CascadeClassifier('Face.xml')
#start cam
cam = cv2.VideoCapture('sample video.mp4')
while True:
    success, frame = cam.read()
    grayimg = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #detect faces
    faceCoordinates = trainedData.detectMultiScale(grayimg)
    for x,y,w,h in faceCoordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (r(0, 256), r(0, 256), r(0, 256)), 2)

    # display image
    cv2.imshow('Result', frame)
    key = cv2.waitKey(1)
    if (key == 81 or key == 113):
        break

print("End of Program")


