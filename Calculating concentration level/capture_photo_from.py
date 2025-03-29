import cv2
import os



# Open the file contains individual videos
student_video_path = 'student_segment'
dir = os.listdir(student_video_path)

for stnt in dir:
    cap = cv2.VideoCapture(student_video_path+"/"+stnt)

    # Check if the video file was opened successfully
    if not cap.isOpened():
        print("Error opening video file.")
        exit()

    base_name, extension = os.path.splitext(stnt)
    frame_number = 0

    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        # Check if the frame was read successfully
        if not ret:
            break

        # Display the frame (optional)
        cv2.imshow('Frame', frame)

        # Save the frame as an image
        # frame_filename = os.path.join('student_id', f'frame_{base_name}.jpg')
        frame_filename = os.path.join('student_id', f'{base_name}.jpg')
        cv2.imwrite(frame_filename, frame)

        frame_number += 1

        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    # Release the video capture object and close the window
    cap.release()
    cv2.destroyAllWindows()