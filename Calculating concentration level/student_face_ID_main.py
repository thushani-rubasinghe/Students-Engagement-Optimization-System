import face_recognition
import cv2
import numpy as np
import os
from gaze_tracking import GazeTracking

real_time_detect_name = "no one detected"

# Folder containing known faces
wn_folder_path = "student_id"
pepl_list = os.listdir(wn_folder_path)
known_face_names = []
known_face_encodings = []

# Load known faces
for pepl in pepl_list:
    print(f"Loading: {wn_folder_path}/{pepl}")

    people_face = face_recognition.load_image_file(f"{wn_folder_path}/{pepl}")

    # Ensure face encodings exist
    encodings = face_recognition.face_encodings(people_face)
    if encodings:
        known_face_encodings.append(encodings[0])
        name, _ = os.path.splitext(pepl)
        known_face_names.append(name)
        print(f"Loaded: {name}")
    else:
        print(f"Warning: No face found in {pepl}, skipping.")

# Initialize variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

gaze = GazeTracking()
webcam = cv2.VideoCapture(0)


def detect_person(frame, annotate=True):
    """
    Detects a person in the frame using face recognition and gaze tracking.
    """
    global process_this_frame, face_locations, face_encodings, face_names, real_time_detect_name

    gaze.refresh(frame)

    # Process every other frame to improve performance
    if process_this_frame:
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        #rgb_small_frame = small_frame[:, :, ::-1]
        rgb_small_frame = small_frame[:, :, ::-1]
        rgb_small_frame = cv2.cvtColor(rgb_small_frame, cv2.COLOR_BGR2RGB)


        # Detect face locations and landmarks
        #face_locations = face_recognition.face_locations(rgb_small_frame)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_landmarks = face_recognition.face_landmarks(rgb_small_frame)

        print("Face Locations:", face_locations)  # Debugging
        print("Face Landmarks:", face_landmarks)  # Debugging

        # Only compute encodings if landmarks are detected
        if face_landmarks:
            face_encodings = face_recognition.face_encodings(rgb_small_frame, known_face_locations=face_locations)
        else:
            face_encodings = []
            print("No face landmarks detected!")

        # Identify faces
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            if matches:
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    # Draw bounding boxes and labels
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw face bounding box
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw name label
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        real_time_detect_name = name

    # Gaze tracking overlay
    frame = gaze.annotated_frame()
    text = ""

    if gaze.is_blinking():
        text = "Blinking"
    elif gaze.is_right():
        text = "Looking right"
    elif gaze.is_left():
        text = "Looking left"
    elif gaze.is_center():
        text = "Looking center"

    cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    cv2.putText(frame, f"Left pupil: {left_pupil}", (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, f"Right pupil: {right_pupil}", (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    # Display the frame
    if annotate:
        cv2.imshow("Video", frame)
        cv2.waitKey(1)

    return frame, real_time_detect_name

# Uncomment for real-time detection
# while True:
#     ret, frame = webcam.read()
#     if not ret:
#         break
#     frame, details = detect_person(frame, annotate=False)
#     print(details)
#     cv2.waitKey(1)
