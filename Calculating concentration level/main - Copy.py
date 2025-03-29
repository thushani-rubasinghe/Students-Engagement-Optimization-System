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
import asyncio

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




# Function create individual videos and identify & name them using the Student photos
@app.route('/create_individual_videos', methods=['GET'])
def create_individual_videos():
    global current_process_status
    current_process_status = "Create Individual Videos"
    student_segmant_path_list = 'student_segment'
    student_segmant_list = os.listdir(student_segmant_path_list)

    for student in student_segmant_list:
        from student_face_ID_main import detect_person
        base_name, extension = os.path.splitext(student)

        student_seg_parth = student_segmant_path_list+'/'+student
        video_capture = cv2.VideoCapture(student_seg_parth)

        time.sleep(1)

        for wash in range(10):
            ret, frame = video_capture.read()
            cv2.waitKey(1)

        for stu_frm in range(20):
            ret, frame = video_capture.read()
            frame,detels = detect_person(frame,anotate=False)
            cv2.imshow('live',frame)
            cv2.waitKey(1)

            if detels == 'Unknown':
                print(detels)
        video_capture.release()
        cv2.destroyAllWindows()
        time.sleep(1)

        source_folder = "student_segment"
        destination_folder = "student_segment_with_ID"
        source_file = os.path.join(source_folder, student)

        if detels != 'Unknown':
            # detels = detels+'.mp4'
            detels = detels + '.mp4'
            destination_file = os.path.join(destination_folder, detels)

        else:
            detels = detels + '.mp4'
            destination_file = os.path.join(destination_folder, detels)

        os.makedirs(destination_folder, exist_ok=True)
        shutil.copy(source_file, destination_file)
        time.sleep(1)

    return "success"

@app.route('/predict_student_status', methods=['GET'])
def predict_student_status(data_frame, details):
    global current_process_status
    current_process_status = "Predict Student Status"

    model = keras.models.load_model('final_analizer.h5')
    predict_data = data_frame
    if details:
        print(predict_data[1:134])
    data_frame = predict_data[1:134]

    data_frame = np.array(data_frame, dtype=float)
    if details:
        print('real_grade - ', predict_data[136])
    data = data_frame.reshape(1, 133)
    if details:
        print(data)

    predictions = model.predict(data)
    accu = np.max(predictions)

    resalt = ['good', 'bad']
    if details:
        print('accuresy - ', round(accu * 100, 2), ' %')
        print('predict behavior - ', resalt[np.argmax(predictions)])
    print('Accuracy : ', round(accu * 100, 2), ' %')

    return resalt[np.argmax(predictions)] , round(accu * 100, 2)


student_repot = []
global student_sequence_number
student_sequence_number = 0

# Function to get the status in Text
@app.route('/write_to_table', methods=['GET'])
def write_to_table(name, good, bad, emotions):
    global current_process_status
    global student_sequence_number
    student_sequence_number = student_sequence_number + 1
    current_process_status = "Preparing Student Reports"
    timestamp = datetime.datetime.now()

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
    total_emotions_count = angry_count + disgust_count + fear_count + happy_count + sad_count + surprise_count + neutral_count
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

    concentration_level = int((max_value/total_emotions_count) * 100)
    good_status = int((good/(good+bad)*100))

    if good_status >= 75:
        action_detected = "Focused"
        attention_state = "High"

    elif good_status >= 50:
        action_detected = "Focused"
        attention_state = "Medium"

    else:
        action_detected = "Not Focused"
        attention_state = "Low"

    student_repot.append([number, name, action_detected, attention_state, angry_count, disgust_count, fear_count, happy_count, sad_count, surprise_count, neutral_count, final_emotion, concentration_level, timestamp])

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
        status = create_individual_videos()

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

                cv2.imshow('live', frame)
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
        csvwriter.writerow(['Number', 'Student', 'Action Detected', 'Attention State', 'Angry Count', 'Disgust Count', 'Fear Count', 'Happy Count', 'Sad Count', 'Surprise Count', 'Neutral Count', 'Final Emotion', 'Concentration Level', 'Time'])
        csvwriter.writerows(student_repot)


# Read data in the CSV file
@app.route('/read_csv', methods=['GET'])
def read_csv():
    global current_process_status
    current_process_status = "Started Reading the CSV File"

    csv_data = []
    with open('student_report.csv') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        line_count = 0

        # Skip CSV headers
        for row in csvreader:
            if line_count == 0:
                line_count += 1
            else:
                csv_data.append(row)

    current_process_status = "Completed Reading the CSV File"
    return csv_data

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Hardcoded user details
        user_username = 'shehani'
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

@app.route('/start_process', methods=['GET', 'POST'])
def start_process():
    obj = {
        "result": read_csv()
    }
    return obj


if __name__ == '__main__':
    app.run(debug=True)
    # create_individual_videos()
    # predict_final_student_status()
    # write_to_csv()
    # read_csv()