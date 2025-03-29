# Import libraries
from flask import Flask, render_template, request, redirect, url_for, flash
from flask import request
from flask_restful import abort
from flask_cors import CORS
import cv2
import os
import numpy as np
import keras
import shutil
import csv
import time
import datetime;


app = Flask(__name__)
app.secret_key = "super secret key"
# app.config["DEBUG"] = True
#
# Allow cross-origin
CORS(app)
cors = CORS(app, resources={
    r"/": {
        "origins": "*"
    }
})

global current_process_status
current_process_status = "Not Yet Started"

# Function return the current progress (status) of the whole process
@app.route('/status', methods=['GET'])
def get_current_process_status():
    global current_process_status
    progress = 0

    if current_process_status == "Not Yet Started":
        progress = 10
    elif current_process_status == "Starting Predict the Student Final Status":
        progress = 20
    elif current_process_status == "Input Data & Create Segments":
        progress = 30
    elif current_process_status == "Create Individual Videos":
        progress = 40
    elif current_process_status == "Tracking Face Emotions":
        progress = 60
    elif current_process_status == "Writing Report Data into the CSV file":
        progress = 75
    elif current_process_status == "Started Reading the CSV File":
        progress = 99


    obj = {
        "result": current_process_status,
        "progress": progress
    }
    return obj

    # obj = {
    #     "status": current_process_status
    # }
    #
    # return obj

# Calling functions in other python files
# def call_external_functions():
#     from video_face_detect import input_and_segment
#     from student_face_ID_main import detect_person
#     from attention_recognition_main import face_movement
#     from imotion_main import student_imo
#
#     file_parth = 'video_input_file/stu2.mp4'
#     input_and_segment(file_parth)

global attendance_time
attendance_time = []

# Function create individual videos and identify & name them using the Student photos
@app.route('/create_individual_videos', methods=['GET'])
def create_individual_videos():
    global current_process_status
    current_process_status = "Create Individual Videos"
    student_segmant_path_list = 'student_segment'
    student_segmant_list = os.listdir(student_segmant_path_list)
    attendance = {}
    global attendance_time

    for student in student_segmant_list:
        from student_face_ID_main import detect_person
        base_name, extension = os.path.splitext(student)

        student_seg_parth = student_segmant_path_list+'/'+student
        video_capture = cv2.VideoCapture(student_seg_parth)

        time.sleep(1)

        for wash in range(10):
            ret, frame = video_capture.read()
            cv2.waitKey(1)

        for stu_frm in range(60):
            ret, frame = video_capture.read()
            frame,detels = detect_person(frame)
            cv2.imshow('live',frame)
            cv2.waitKey(1)

            if detels == 'Unknown':
                print(detels)

                # Mark attendance for this student in minutes
            attendance[detels] = attendance.get(detels, 0) + 1 / 60
            # attendance[detels] = attendance.get((detels, 0) + 1 / 60)

        video_capture.release()
        cv2.destroyAllWindows()
        time.sleep(1)

        source_folder = "student_segment"
        destination_folder = "student_segment_with_ID"
        source_file = os.path.join(source_folder, student)

        if detels != 'Unknown':
            # details = details+'.mp4'
            detels = detels + '.mp4'
            destination_file = os.path.join(destination_folder, detels)

        else:
            detels = detels + '.mp4'
            destination_file = os.path.join(destination_folder, detels)

        os.makedirs(destination_folder, exist_ok=True)
        shutil.copy(source_file, destination_file)
        time.sleep(1)

        # Print the attendance report in minutes
        print("\nAttendance Report (in minutes):")
        for student, time_minutes in attendance.items():
            print(f"{student}: {time_minutes:.2f} minutes")
        attendance_time.append(round(time_minutes, 2))

        # Calculate and print the total attendance in minutes
        total_attendance_minutes = sum(attendance.values())
        print(f"Total Attendance: {total_attendance_minutes:.2f} minutes")

        # Calculate the attendance percentage
        total_students = len(student_segmant_list)
        attendance_percentage = (total_attendance_minutes / (total_students * 20)) * 100
        print(f"Attendance Percentage: {attendance_percentage:.2f}%")

        # Create a CSV file to store attendance details
        csv_filename = "attendance_details.csv"

        with open(csv_filename, mode='w', newline='') as csvfile:
            fieldnames = ['Student', 'Attendance (hours:minutes)']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Write the header row
            writer.writeheader()

            # Write attendance details to the CSV file with formatting
            for student, time_minutes in attendance.items():
                hours = int(time_minutes // 60)
                minutes = int(time_minutes % 60)
                formatted_time = f"{hours:02d}:{minutes:02d}"
                writer.writerow({'Student': student, 'Attendance (hours:minutes)': formatted_time})

        print(f"Attendance details have been saved to '{csv_filename}'.")
        print(attendance_time)

    return "success"

@app.route('/predict_student_status', methods=['GET'])
def predict_student_status(data_frame, detels):
    global current_process_status
    current_process_status = "Predict Student Status"

    model = keras.models.load_model('final_analizer.h5')
    predict_data = data_frame
    if detels:
        print(predict_data[1:134])
    data_frame = predict_data[1:134]

    data_frame = np.array(data_frame, dtype=float)
    if detels:
        print('real_grade - ', predict_data[136])
    data = data_frame.reshape(1, 133)
    if detels:
        print(data)

    predictions = model.predict(data)
    accu = np.max(predictions)

    resalt = ['good', 'bad']
    if detels:
        print('accuresy - ', round(accu * 100, 2), ' %')
        print('predict behavior - ', resalt[np.argmax(predictions)])
    print('Accuracy : ', round(accu * 100, 2), ' %')

    return resalt[np.argmax(predictions)] , round(accu * 100, 2)


student_repot = []
global student_sequence_number
# global current_timestamp
# current_timestamp = datetime.datetime.now()
# current_timestamp = datetime.datetime.strptime(current_timestamp, "%Y-%m-%d %H:%M:%S")
student_sequence_number = 0

# Function to get the status in Text
@app.route('/write_to_table', methods=['GET'])
def write_to_table(name, good, bad, emotions):
    global current_timestamp
    global current_process_status
    global student_sequence_number
    student_sequence_number = student_sequence_number + 1
    current_process_status = "Preparing Student Reports"
    # starting_timestamp = datetime.datetime.now()
    # starting_timestamp = datetime.strptime(starting_timestamp, "%Y-%m-%d %H:%M:%S")

    # Get the time difference of the individual student process
    # if student_sequence_number == 1:
    #     student_timestamp = current_timestamp - starting_timestamp
    
    # else:
    #     student_timestamp = current_timestamp - starting_timestamp

    # current_timestamp = starting_timestamp


    # Get the current student attend time and decide the attendance
    attend_time = attendance_time[student_sequence_number - 1]
    attend_time = attend_time * 10
    student_attend_time_in_seconds = attend_time * 10

    # If the student attendance time is more than 30 seconds
    if attend_time > 3.0:
        attendance = "Present"
    else:
        attendance = "Absent"

    # Get the count of emotions and other table data
    number = student_sequence_number
    angry_count = emotions.count('angry')
    disgust_count = emotions.count('disgust')
    fear_count = emotions.count('fear')
    happy_count = emotions.count('happy')
    sad_count = emotions.count('sad')
    surprise_count = emotions.count('surprise')
    neutral_count = emotions.count('neutral')

    emotions_count_array = [angry_count, disgust_count, fear_count, happy_count, sad_count, surprise_count, neutral_count]
    max_value = 0

    for i in emotions_count_array:
        if max_value < i:
            max_value = i

    if max_value == angry_count:
        final_emotion = "Angry"
    elif max_value == disgust_count:
        final_emotion = "Disgust"
    elif max_value == fear_count:
        final_emotion = "Fear"
    elif max_value == happy_count:
        final_emotion = "Happy"
    elif max_value == sad_count:
        final_emotion = "Sad"
    elif max_value == surprise_count:
        final_emotion = "Surprise"
    elif max_value == neutral_count:
        final_emotion = "Neutral"

    concentration_level = int((good/(good+bad)*100))

    # Get the attention status by final emotion
    if final_emotion == "Neutral" or final_emotion == "Happy":
        attention_state = "High"

    elif final_emotion == "Surprise" or final_emotion == "Fear":
        attention_state = "Medium"

    else:
        attention_state = "Low"


    # Get the student Action status by the concentration level
    if concentration_level >= 75:
        action_detected = "Focused"

    elif concentration_level >= 50:
        action_detected = "Focused"

    else:
        action_detected = "Not Focused"
        

    student_repot.append([number, name, action_detected, attention_state, angry_count, disgust_count, fear_count, happy_count, sad_count, surprise_count, neutral_count, final_emotion, concentration_level, student_attend_time_in_seconds, attendance])

    obj = {
        "status": "success"
    }

    return obj


@app.route('/predict_final_student_status', methods=['GET', 'POST'])
def predict_final_student_status():
    if request.method == 'POST':
        video_file_path = request.form['video_file_path']
        global current_process_status
        current_process_status = "Starting Predict the Student Final Status"

        from video_face_detect import input_and_segment
        from student_face_ID_main import detect_person
        from attention_recognition_main import face_movement
        from imotion_main import student_imo

        file_parth = 'video_input_file/' + video_file_path
        current_process_status = "Input Data & Create Segments"

        # Split videos by students (face) and save them as 'student 0, student 1 & so on...'
        input_and_segment(file_parth)

        # Naming split videos by actual student names (eg: Sasindu.mp4)
        create_individual_videos()

        student_segment_file_path = 'student_segment_with_ID'
        student_segment_list = os.listdir(student_segment_file_path)

        for student in student_segment_list:
            print(student)

            student_name, extension = os.path.splitext(student)
            student_attention = []
            student_emotion = []

            student_seg_id_parth = student_segment_file_path + '/' + student
            print(student_seg_id_parth)
            cap = cv2.VideoCapture(student_seg_id_parth)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            time.sleep(1)

            for x in range(total_frames):

                ret, frame = cap.read()

                try:
                    # Tracking face movements
                    current_process_status = "Track Face Movements"
                    frame, rxtract_data_attention = face_movement(frame, anotation=True)
                    current_process_status = "Tracking Face Emotions"

                    # Tracking student emotions
                    frame, emo_data_extract = student_imo(frame, detels=True)
                    student_emotion.append(emo_data_extract[7])

                    # frame, rxtract_data_attention = face_movement(frame, anotation=False)
                    # frame, emo_data_extract = student_imo(frame, detels=False)

                    if rxtract_data_attention.shape == (129,) and emo_data_extract.shape == (8,):
                        student_extract_data = np.concatenate((rxtract_data_attention, emo_data_extract), axis=0)

                        #print(student_extract_data)
                        student_status, acc = predict_student_status(student_extract_data, detels=False)
                        # student_status = 'good' # model()
                        print('predict from model - status :'+student_status,'   -   accuresy - ',acc)

                except:
                    student_status = 'good'

                student_attention.append(student_status)

                # cv2.imshow('live', frame)
                cv2.waitKey(1)

            cap.release()
            cv2.destroyAllWindows()

            num_goods = student_attention.count('good')
            num_bads = student_attention.count('bad')

            # Calling count_emotions function to get the total count of emotions
            # count_emotions(student_emotion)

            # Calling write_to_table function to generate an array for final results
            write_to_table(student_name, num_goods, num_bads, student_emotion)

        current_process_status = "Stopped Predicting the Student Final Status"

        obj = {
            "status": "success"
        }

    write_to_csv()
    time.sleep(5)
    csv_data = read_csv()
    time.sleep(5)

    return render_template('result.html', table_data=csv_data)


# Writing to the csv file
@app.route('/write_to_csv', methods=['GET'])
def write_to_csv():
    global current_process_status
    current_process_status = "Writing Report Data into the CSV file"

    with open("student_report.csv", 'w', newline='') as csvfile:
        # Creating a csv writer object
        csvwriter = csv.writer(csvfile)

        # Writing the contents of the list to the CSV
        csvwriter.writerow(['Number', 'Student', 'Action Detected', 'Attention State', 'Angry Count', 'Disgust Count', 'Fear Count', 'Happy Count', 'Sad Count', 'Surprise Count', 'Neutral Count', 'Final Emotion', 'Concentration Level', 'Time', 'Attendance'])
        csvwriter.writerows(student_repot)

global total_emotions_counts
total_emotions_counts = []

# Read data in the CSV file
@app.route('/read_csv', methods=['GET'])
def read_csv():
    global current_process_status
    global total_emotions_counts
    current_process_status = "Started Reading the CSV File"

    csv_data = []
    with open('student_report.csv') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        line_count = 0
        total_angry_count = 0
        total_disgust_count = 0
        total_fear_count = 0
        total_happy_count = 0
        total_sad_count = 0
        total_surprise_count = 0
        total_neutral_count = 0

        # Skip CSV headers
        for row in csvreader:
            if line_count == 0:
                line_count += 1
            else:
                csv_data.append(row)
                total_angry_count = total_angry_count + int(row[4])
                total_disgust_count = total_disgust_count + int(row[5])
                total_fear_count = total_fear_count + int(row[6])
                total_happy_count = total_happy_count + int(row[7])
                total_sad_count = total_sad_count + int(row[8])
                total_surprise_count = total_surprise_count + int(row[9])
                total_neutral_count = total_neutral_count + int(row[10])
    
        # Get the total count of each emotions
        total_emotions_counts = [total_angry_count, total_disgust_count, total_fear_count, total_happy_count, total_sad_count, total_surprise_count, total_neutral_count]
        print(total_emotions_counts)

    current_process_status = "Completed Reading the CSV File"
    return csv_data

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Hardcoded user details
        user_username = 'thushani'
        user_password = '12345'

        email = request.form.get('email')
        password = request.form.get('password')

        if email != user_username:
            message = 'Entered username is not found'
            return render_template('login.html', message=message)
        elif password != user_password:
            message = 'Entered password is incorrect'
            return render_template('login.html', message=message)
        else:
            message = 'Login successfully'
            return render_template('home.html', message=message)

    return render_template('login.html')

# Function to go to the Home page
@app.route('/home', methods=['GET'])
def navigate_to_home():
    return render_template('home.html')

# Function to go to the Result Page
@app.route('/result', methods=['GET', 'POST'])
def navigate_to_results():
    csv_data = read_csv()

    return render_template('result.html', table_data=csv_data)


# Function to feed data to the Bar chart in the 'result.html' page
@app.route('/start_process', methods=['GET', 'POST'])
def start_process():
    global total_emotions_counts

    obj = {
        "result": read_csv(),
        "total_emotions_counts": total_emotions_counts
    }
    print(obj)
    return obj


# Function to increase the resolution of videos
@app.route('/resolution', methods=['GET', 'POST'])
def video_resolution():
    if request.method == 'GET':
        return render_template('resolution.html')

if __name__ == '__main__':
    app.run(debug=True)
    # create_individual_videos()
    # predict_final_student_status()
    # write_to_csv()
    # read_csv()