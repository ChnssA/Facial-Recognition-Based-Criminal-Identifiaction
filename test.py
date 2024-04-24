# pip install opencv-python==4.5.2

import cv2

import time

video = cv2.VideoCapture(0)

facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("Trainer.yml")

name_list = ["","Vishnu","Shruti_Sagar","neeraj","Sudhanshu_jha" ]
age_list=["","20","21","21","20"]
crime_list = ["","Robber","Hacker","theif","_"]



while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        serial, conf = recognizer.predict(gray[y:y + h, x:x + w])

        if conf > 50:

            cv2.putText(frame,"Name:"+ name_list[serial], (x, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 127), 2)
            cv2.putText(frame,"Age:"+ age_list[serial], (x, y - 33), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 127), 2)
            cv2.putText(frame,"Crime:"+ crime_list[serial], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 127), 2)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),1)
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)
            cv2.putText(frame, "Unknown_face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (55, 55, 255), 2)
    frame = cv2.resize(frame, (640, 480))
    cv2.imshow("frame",frame)


    k = cv2.waitKey(1)

    if k == ord('o') and conf >50 :

        time.sleep(10)

    if k == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
