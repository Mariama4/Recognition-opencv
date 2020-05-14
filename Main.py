import os
from PIL import Image
import cv2
import numpy as np



with open("names.txt") as file:
    str_names = [row.strip() for row in file]

f = open('names.txt', 'a')

def face_dataset():

    # For each person, enter one numeric face id


    count_samples = input('\n Enter count of face sample ==>  ')

    if choice == '4':
        cam = cv2.VideoCapture(0)
    else:
        # Initialize and start realtime video capture
        cam = cv2.VideoCapture(url_for_training)

    cam.set(3, 640)  # set video width
    cam.set(4, 480)  # set video height

    face_detector = cv2.CascadeClassifier('trainer/haarcascade_frontalface_default.xml')  # face

    print("\n [INFO] Initializing face capture. Wait ...")

    # Initialize individual sampling face count
    count = 0

    while (True):

        ret, img = cam.read()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30)
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            count += 1

            # Save the captured image into the datasets folder
            cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg",
                        gray[y:y + h, x:x + w])

            cv2.imshow('image', img)

        k = cv2.waitKey(100) & 0xff  # Press 'ESC' for exiting video
        if k == 27:
            break
        elif count >= int(count_samples):  # Take N face sample and stop video
            break

    # Do a bit of cleanup
    print("\n [INFO] Exiting def and start training.")

    # starting face_training

    cam.release()
    cv2.destroyAllWindows()
    face_training()





def face_training():

    # Path for face image database
    path = 'dataset'

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier("trainer/haarcascade_frontalface_default.xml")

    # function to get the images and label data
    def getImagesAndLabels(path):

        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        faceSamples = []
        ids = []

        for imagePath in imagePaths:

            PIL_img = Image.open(imagePath).convert('L')  # convert it to grayscale
            img_numpy = np.array(PIL_img, 'uint8')

            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = detector.detectMultiScale(img_numpy)

            for (x, y, w, h) in faces:
                faceSamples.append(img_numpy[y:y + h, x:x + w])
                ids.append(id)

        return faceSamples, ids

    print("\n [INFO] Training faces. It will take a few seconds. Wait ...")
    faces, ids = getImagesAndLabels(path)
    recognizer.train(faces, np.array(ids))

    # Save the model into trainer/trainer.yml
    recognizer.write('trainer/trainer.yml')  # recognizer.save() worked on Mac, but not on Pi

    # Print the numer of faces trained and end program
    print("\n [INFO] {0} faces trained.".format(len(np.unique(ids))))




def face_recognition():

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer/trainer.yml')
    cascadePath = "trainer/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath);

    font = cv2.FONT_HERSHEY_SIMPLEX

    # iniciate id counter
    id = 0

    names = str_names

    if ((choice == '1') or (choice == '2') or (choice == '4')):
        #append new names in txt
        f.write("%s\n" % id_name)


    if choice == '4':
        cam = cv2.VideoCapture(0)
    else:
        # Initialize and start realtime video capture
        cam = cv2.VideoCapture(url)

    cam.set(3, 640)  # set video widht
    cam.set(4, 480)  # set video height

    # Define min window size to be recognized as a face
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)

    while True:
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
                confidence = "  {0}%".format(round(100 - confidence))
                print('Есть совпадение! '+ str(confidence))

            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))
                print('Нет совпадений!')

            cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

        cv2.imshow('camera', img)

        k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
        if k == 27:
            break # termination async callback "because of this"

    # Do a bit of cleanup
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()


def CamTest():
    cap = cv2.VideoCapture(0)

    while (True):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow('frame', frame)
        cv2.imshow('gray', gray)
        k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
        if k == 27:
            break # termination async callback "because of this"

    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    print(' Halo! '
          '\n Писал все на английском'
          '\n Before using the second file, you need to add yourself to the database and train the model!'
          '\n What do you want? \n 1 - Create dataset and training \n 2 - Create dataset and test it on webcam'
          '\n 3 - Test on  ip or webcam'
          '\n 4 - Create dataset, training and test on WebCam'
          '\n Warning! Now the model can check 1 thread: Ip cam or Webcam'
          '\n Check the second file to check the 2 threads')
    choice = input('\n ==>  ')
    if choice == '1':
        _ = input('Look at file or webcam? 0 - webcam 1 - file ==>  ')
        if _ == '0':
            url_for_training = int(_)
        elif _ == '1':
            print('\n Example "0" or ip adress: http://192.168.0.100:8080/video ')
            url_for_training = (input('\n Input source ==>  '))
        else:
            print('Fail')
            raise SystemExit
        print('Names related to ids: example ==> Georgy: id=1, You: id=2  etc. (None = 0)')
        print('Occupied: ' + str(str_names))
        face_id = input('\n Enter user id end press  ==>  ')
        id_name = input('\n Enter user NAME end press enter  ==>  ')
        face_dataset()
        f.close()
    elif choice == '2':
        _ = input('Look at file or webcam? 0 - webcam 1 - file ==>  ')
        if _ == '0':
            url = int(_)
            url_for_training = url
        elif _ == '1':
            print('\n Example "0" or ip adress: http://192.168.0.100:8080/video ')
            url = (input('\n Input source ==>  '))
            url_for_training = url
        else:
            print('Fail')
            raise SystemExit
        print('Names related to ids: example ==> Georgy: id=1, You: id=2  etc. (None = 0)')
        print('Occupied: ' + str(str_names))
        face_id = input('\n Enter user id end press  ==>  ')
        id_name = input('\n Enter user NAME end press enter  ==>  ')
        face_dataset()
        face_recognition()
        f.close()
    elif choice == '3':
        _ = input('Look at file or webcam? 0 - webcam 1 - file ==>  ')
        if _ == '0':
            url = int(_)
        elif _ == '1':
            print('\n Example "0" or ip adress: http://192.168.0.100:8080/video ')
            url = (input('\n Input source ==>  '))
        else:
            print('Fail')
            raise SystemExit
        face_recognition()
        f.close()
    elif choice == '4':
        url = 0
        print('Names related to ids: example ==> Georgy: id=1, You: id=2  etc. (None = 0)')
        print('Occupied: ' + str(str_names))
        face_id = input('\n Enter user ID end press enter  ==>  ')
        id_name = input('\n Enter user NAME end press enter  ==>  ')
        print('Before we start, we need to test the camera ==>  ')
        print('Press Esc, if all ok ==>  ')
        CamTest()
        face_dataset()
        face_recognition()
        f.close()

    else:
        print('Fail')
        raise SystemExit