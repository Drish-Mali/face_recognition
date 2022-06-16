# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from mtcnn.mtcnn import MTCNN
import cv2
import model
from model import result

def run():

    print(cv2.__version__)
    detector = MTCNN()

    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()
        '''face_cas=cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces=face_cas.detectMultiScale(image=gray,scaleFactor=1.2,minNeighbors=3)'''
        faces = detector.detect_faces(img)
        if len(faces)!=0:
            for i in faces:
                bounding_box = i['box']
                face = img[bounding_box[1]:(bounding_box[1] + bounding_box[3]),
                       bounding_box[0]:(bounding_box[0] + bounding_box[2])]
                cv2.rectangle(img,
                      (bounding_box[0], bounding_box[1]),
                      (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                      (0, 155, 255),2)
                resize=cv2.resize(face,(150,150),interpolation = cv2.INTER_LINEAR)

                img=cv2.putText(img,result(resize),(bounding_box[0],bounding_box[1]),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),1)
        '''for (x,y,w,h) in faces:
            cv2.rectangle(img,pt1=(x,y),pt2=(x+w,y+h),color=(255,0,0))'''


        cv2.imshow('pic', img)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            cap.release()
            break
    cv2.destroyAllWindows()

run()
