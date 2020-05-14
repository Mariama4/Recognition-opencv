import datetime
import smtplib
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import cv2


def mail_report(name, img): # I think it's clear

    msg = MIMEMultipart()

    now = datetime.datetime.now()

    msg['Subject'] = ('Report ' + now.strftime("%d-%m-%Y %H:%M"))
    msg['From'] = 'fvone@inbox.ru'

    part = MIMEText(str(name) + ' at ' + now.strftime('%d-%m-%Y %H:%M'))
    msg.attach(part)

    smtpObj = smtplib.SMTP('smtp.mail.ru', 587)

    smtpObj.starttls()

    smtpObj.login('fvone@inbox.ru', 'pAYITp$puj34')

    cv2.imwrite(('database/' + (str(name) + now.strftime('%d-%H-%M') + '.jpg')), img)
    part = MIMEImage(open(str('database/' + str(name) + now.strftime('%d-%H-%M') + '.jpg'), 'rb').read())
    msg.attach(part)

    smtpObj.sendmail(msg['From'], [report[0]], msg.as_string())

    smtpObj.quit()

def recognition(cam):

    ret, img = cam.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )

    for (x, y, w, h) in faces:

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

        # Check if confidence is less them 100 ==> "0" is perfect match
        if (confidence < 100):
            id = names[id]
            if (confidence <= 40):
                mail_report(id, img) # Warninng! There may be lags if there are a large number of matches!
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

    cv2.imshow(str(cam), img)

recognizer = cv2.face.LBPHFaceRecognizer_create() # loading the model
recognizer.read('trainer/trainer.yml')
cascadePath = "trainer/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_SIMPLEX  #text settings

id = 0

with open('report.txt') as file:
    report = [row.strip() for row in file]

with open('names.txt') as file:
    names = [row.strip() for row in file] # loading names from .txt file

with open('source.txt') as file:
    source = [row.strip() for row in file]
if source[0] == "0":
    cam = cv2.VideoCapture(0) # source 1
else:
    cam = cv2.VideoCapture(0)  # source 1
    cam1 = cv2.VideoCapture(str(source[0])) # source 2

    cam1.set(3, 640)  # set video widht for souce 2
    cam1.set(4, 480)  # set video height

    minW1 = 0.1 * cam1.get(3)  # for source 2, because they can they may have different extensions
    minH1 = 0.1 * cam1.get(4)

cam.set(3, 640)  # set video widht for source 1
cam.set(4, 480)  # set video height



# Define min window size to be recognized as a face
minW = 0.1 * cam.get(3) #for source 1
minH = 0.1 * cam.get(4)



while True:
    if source[0] == "0":
        recognition(cam) # rec on cam 1
    else:
        recognition(cam)  # rec on cam 1
        recognition(cam1) # rec on cam 2
    k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
    if k == 27:
        break # termination async callback "because of this"

print("\n [INFO] Exiting Program and cleanup stuff")
if source[0] == "0":
    cam.release()
else:
    cam.release()
    cam1.release()
cv2.destroyAllWindows()