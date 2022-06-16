from mtcnn.mtcnn import MTCNN
import cv2
import os

def input_data(name):
    detector = MTCNN()
    count = 1
    train_dir = "C:\\Users\\deepa\\PycharmProjects\\opencv\\dataset\\train"
    test_dir="C:\\Users\\deepa\\PycharmProjects\\opencv\\dataset\\test"
    path = os.path.join(test_dir, name)
    os.mkdir(path)
    os.chdir(train_dir)
    path=os.path.join(name)
    os.mkdir(path)
    os.chdir(path)
    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()

        try:
            faces = detector.detect_faces(img)
        except:
            print("finsh")
            break
        #print((faces))

        # faces = detector.detect_faces(image)
        if len(faces) != 0:
            if count > 75:
                path = os.path.join(test_dir, name)
                os.chdir(path)
            for i in faces:
                bounding_box = i['box']

                face = img[bounding_box[1]:(bounding_box[1] + bounding_box[3]),bounding_box[0]:(bounding_box[0] + bounding_box[2])]
                cv2.rectangle(img,
                              (bounding_box[0], bounding_box[1]),
                              (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                              (0, 155, 255),
                              2)
                if(count<=100):


                    fname=name+str(count)+".jpg"


                    cv2.imwrite(fname, face)
                    print(count)
                    count=count+1

                else:
                    cap.release()
                    cv2.destroyAllWindows()
                    break
        '''for (x,y,w,h) in faces:
            cv2.rectangle(img,pt1=(x,y),pt2=(x+w,y+h),color=(255,0,0))'''

        cv2.imshow('pic', img)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            cap.release()
            break
    cv2.destroyAllWindows()
#input_data('rashila')

