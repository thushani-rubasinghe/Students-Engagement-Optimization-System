import face_recognition
import cv2
import numpy as np
import os
import threading

real_time_detect_name = 'no one detect'

# Below function will get the list of items (images) inside the given folder location (path)
wn_folder_parth ='student_id'
pepl_list = os.listdir(wn_folder_parth)
known_face_names = []
known_face_encodings = []

# Below 'for-loop' loops number of images times in the folder location
for pepl in pepl_list:

    print(wn_folder_parth+'/'+pepl)

    people_face = face_recognition.load_image_file(wn_folder_parth+'/'+pepl)
    people_face_encoding = face_recognition.face_encodings(people_face)[0]
    known_face_encodings.append(people_face_encoding)

    # Split the image name by "." and extract the student name
    name, format = pepl.split(".")
    known_face_names.append(name)
    print('Entered peoples name - ',name)

    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True


def detect_person(frame,anotate=True):
    global process_this_frame,face_locations , face_names,face_encodings,real_time_detect_name

    # Only process every other frame of video to save time
    if process_this_frame:
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        real_time_detect_name = name

    # Display the resulting image
    if anotate:
        cv2.imshow('Video', frame)
        cv2.waitKey(1)
    return frame , real_time_detect_name

# video_capture = cv2.VideoCapture('Thanks Obama.mp4')
#
# while True:
#
#     ret, frame = video_capture.read()
#     frame,detels = detect_person(frame,anotate=False)
#     #cv2.imshow('liive',frame)
#     print(detels)
#     cv2.waitKey(1)





