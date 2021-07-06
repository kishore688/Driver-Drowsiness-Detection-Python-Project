import cv2
from keras.models import load_model
import os
import numpy as np
from pygame import mixer
import time

mixer.init()
sound = mixer.Sound(
    'C:\\Users\\kishore sai\\Downloads\\Compressed\\Drowsiness detection\\Drowsiness detection\\alarm.wav')

face = cv2.CascadeClassifier('C:\\Users\\kishore sai\\Downloads\\Compressed\\Drowsiness detection\\Drowsiness detection\\haar cascade files\\haarcascade_frontalface_alt.xml')
left_eye = cv2.CascadeClassifier('C:\\Users\\kishore sai\\Downloads\\Compressed\\Drowsiness detection\\Drowsiness detection\\haar cascade files\\haarcascade_lefteye_2splits.xml')
right_eye = cv2.CascadeClassifier('C:\\Users\\kishore sai\\Downloads\\Compressed\\Drowsiness detection\\Drowsiness detection\\haar cascade files\\haarcascade_righteye_2splits.xml')

st = ['open', 'closed']

model = load_model('C:\\Users\\kishore sai\\Downloads\\Compressed\\Drowsiness detection\\Drowsiness detection\\models\\cnnCat2.h5')
path = os.getcwd()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
score = 0
count = 0
thicc = 2
lpred = [99]
rpred = [99]

while (True):
    tr, frame = cap.read()
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
    l_eye = left_eye.detectMultiScale(gray)
    r_eye = right_eye.detectMultiScale(gray)
    cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )
    for (x, y, w, h) in r_eye:
        reye = frame[y:y + h, x:x + w]
        count = count + 1
        reye = cv2.cvtColor(reye, cv2.COLOR_BGR2GRAY)
        reye = cv2.resize(reye, (24, 24))
        reye = reye / 255
        reye = reye.reshape(24, 24, -1)
        reye = np.expand_dims(reye, axis=0)
        rpred = model.predict_classes(reye)
        if rpred[0] == 1:
            st = 'open'
        if rpred[0] == 0:
            st = 'closed'
        break

    for (x, y, w, h) in l_eye:
        leye = frame[y:y + h, x:x + w]
        count = count + 1
        leye = cv2.cvtColor(leye, cv2.COLOR_BGR2GRAY)
        leye = cv2.resize(leye, (24, 24))
        leye = leye / 255
        leye = leye.reshape(24, 24, -1)
        leye = np.expand_dims(leye, axis=0)
        lpred = model.predict_classes(leye)
        if lpred[0] == 1:
            st = 'open'
        if lpred[0] == 0:
            st = 'closed'
        break

    if rpred[0] == 0 and lpred[0] == 0:
        score = score + 1
        cv2.putText(frame, "closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        score = score - 1
        cv2.putText(frame, "open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    if score < 0:
        score = 0
    cv2.putText(frame, "Score:" + str(score), (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    if score > 15:
        cv2.imwrite(os.path.join('image.jpg'), frame)
        try:
            sound.play()
        except:
            pass
        if thicc < 16:
            thicc = thicc + 2
        else:
            thicc = thicc - 2
            if thicc < 0:
                thicc = 2
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
cap.release()
cap.destroyAllWindows()