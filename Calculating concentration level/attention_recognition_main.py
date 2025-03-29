import cv2
import dlib
import numpy as np

#cap = cv2.VideoCapture('Thanks Obama.mp4')
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def face_movement(frame, anotation = True):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = hog_face_detector(gray)
    face_keypoints_extract = []

    for face in faces:
        if faces[0] == face:
            face_landmarks = dlib_facelandmark(gray, face)
            #print(face_landmarks)
            face_keypoints_extract = []
            face_keypoints_x = []
            face_keypoints_y = []

            for n in range(0, 64):
                x = face_landmarks.part(n).x
                y = face_landmarks.part(n).y

                if n == n:
                    cv2.circle(frame, (x, y), 1, (0, 255, 255), 1)
                face_keypoints_x.append(x)
                face_keypoints_y.append(y)

                if n == 0 or n == 16 or n == 8 or n == 27 :

                    cv2.circle(frame, (x, y), 5, (0, 255, 0), 1)

            face_vertical_l = face_landmarks.part(8).y - face_landmarks.part(27).y
            face_horizontal_l = face_landmarks.part(16).x - face_landmarks.part(0).x

            face_keypoints_x = (np.array(face_keypoints_x)- face_landmarks.part(0).x)/face_horizontal_l
            face_keypoints_y = (np.array(face_keypoints_y) - ((face_landmarks.part(19).y + face_landmarks.part(24).y)/2))/face_vertical_l
            face_keypoints_extract = np.concatenate((face_keypoints_x, face_keypoints_y), axis=0)

            x16 = face_landmarks.part(15).x
            x2  = face_landmarks.part(1).x
            x_noz = face_landmarks.part(33).x

            logic = ((x_noz - x2) / (x16 - x2))*100
            face_keypoints_extract = np.concatenate((face_keypoints_extract, np.array([logic])), axis=0)

            if anotation:
                print(face_keypoints_extract)

    if anotation:
        cv2.imshow("Face Landmarks", frame)

    cv2.waitKey(1)
    return frame,face_keypoints_extract

# cap.release()
# cv2.destroyAllWindows()

# while True:
#     _, frame = cap.read()
#
#     fac_box = [100,200, 100,200]    ## x max , x min , y max , y min
#
#     rxtract_data_attention = face_movement(frame,fac_box,anotation=False)
#
#     print(rxtract_data_attention)
#
#     cv2.imshow('live',frame)
#     cv2.waitKey(1)

