import cv2 
import numpy as np 
import dlib

#0 for internal Webcam
cap = cv2.VideoCapture(0)

#Detecttor for face
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
while True:
    #Get the frame
    _, frame = cap.read()
    #Gray only has one color channel- easier for cpu to compute
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detect on gray frame
    faces = detector(gray)
    #Get coordinates for rectangle around face
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        #Draw Rectangle, 3 = thickness
       # cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 3)

        #DPredict landmarks in face area
        landmarks = predictor(gray, face)

        #Show eye landmarks
        for n in range (36, 48):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            #cv2.circle(image, center_coordinates, radius, color, thickness)
            cv2.circle(frame, (x,y), 2,(0, 0, 255))

    #Show the frame
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    #Escape char
    if key ==27:
        break
'''
Various:
#Get co-ordinates for specific landmark
x = landmarks.part(12).x
y = landmarks.part(12).y
print(x,y)

#Draw Rectangle, 3 = thickness
cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 3)
'''