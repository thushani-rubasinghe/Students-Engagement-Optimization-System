# import cv2
# import numpy as np
# import os.path as osp
# import glob
from flask import Flask, render_template, request, redirect, url_for, flash
from flask import request
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch
import time


app = Flask(__name__)

def start_process(filename):
    # stu2 320x240.avi
    print("start_process function is starting")
    increase_resolution(filename)

# input_path = 'video_input_file/stu2 320x240.avi'
# output_path = 'enhanced_video_file/stu2 320x240 gpu result.avi'
# #output_path = 'output_video_file/student 0 360p gpu upp.mp4'
#
# x_power = 4
#
#
# model_path = 'models/RRDB_ESRGAN_x4.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
# # device = torch.device('cpu')  # if you want to run on CPU, change 'cuda' -> cpu
# #device = torch.device('cpu')
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
# test_img_folder = 'LR/*'
#
# model = arch.RRDBNet(3, 3, 64, 23, gc=32)
# model.load_state_dict(torch.load(model_path), strict=True)
# model.eval()
# model = model.to(device)
#
# print('Model path {:s}. \nTesting...'.format(model_path))
#
#
# video_capture = cv2.VideoCapture(input_path)
# frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = video_capture.get(cv2.CAP_PROP_FPS)
# total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
# output_video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width*x_power, frame_height*x_power))


def opencv_upscale(img):

    global frame_width , frame_height , x_power

    scale_percent = 100*x_power  # percent of original size
    width = int(frame_width * scale_percent / 100)
    height = int(frame_height * scale_percent / 100)
    dim = (width, height)

    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    return resized



def SRGAN_superResolutuin(img):

    global model

    #print('in - ',img.shape)

    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    with torch.no_grad():

        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()

    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()
    output = np.array(output, dtype=np.uint8)

    # cv2.imshow('img',output)
    # cv2.waitKey(0)
    return output


# frame_count = 0
# t_loop_time = 0.00

def increase_resolution(filename):
    print("increase_resolution function is starting")
    # input_path = 'video_input_file/stu2 320x240.avi'
    input_path = 'video_input_file/' + filename
    output_path = 'enhanced_video_file/stu2 320x240 gpu result.avi'
    # output_path = 'output_video_file/student 0 360p gpu upp.mp4'

    x_power = 4

    model_path = 'models/RRDB_ESRGAN_x4.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
    # device = torch.device('cpu')  # if you want to run on CPU, change 'cuda' -> cpu
    # device = torch.device('cpu')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_img_folder = 'LR/*'

    model = arch.RRDBNet(3, 3, 64, 23, gc=32)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)

    print('Model path {:s}. \nTesting...'.format(model_path))

    video_capture = cv2.VideoCapture(input_path)
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    output_video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                   (frame_width * x_power, frame_height * x_power))

    frame_count = 0
    t_loop_time = 0.00

    while True:



        start_time = time.time()

        ret, img = video_capture.read()
        if not ret:
            break

        #print('main - ',frame.shape)


        #s_frame = opencv_upscale(frame)
        # s_frame = SRGAN_superResolutuin(frame)

        img = img * 1.0 / 255
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_LR = img.unsqueeze(0)
        img_LR = img_LR.to(device)

        with torch.no_grad():

            output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()

        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round()
        output = np.array(output, dtype=np.uint8)
        s_frame = output

        # cv2.imshow('img',output)
        # cv2.waitKey(0)
        # return output



        output_video.write(s_frame)
        frame_count = frame_count + 1
        #print(frame_count)

        # cv2.waitKey(1)

        loop_time = time.time() - start_time
        ast_finish_time = int(loop_time*(total_frames-frame_count))
        t_loop_time = t_loop_time + loop_time
        total_process_time = int(loop_time*total_frames)

        ast_time_seconds = ast_finish_time
        ast_days = ast_time_seconds // 86400
        ast_time_seconds %= 86400
        ast_hours = ast_time_seconds // 3600
        ast_time_seconds %= 3600
        ast_minutes = ast_time_seconds // 60
        ast_seconds = ast_time_seconds % 60
        formatted_time = "{:02d}:{:02d}:{:02d}:{:02d}".format(ast_days, ast_hours, ast_minutes, ast_seconds)

        total_time_seconds = total_process_time
        total_days = total_time_seconds // 86400
        total_time_seconds %= 86400
        total_hours = total_time_seconds // 3600
        total_time_seconds %= 3600
        total_minutes = total_time_seconds // 60
        total_seconds = total_time_seconds % 60
        total_process_formatted_time = "{:02d}:{:02d}:{:02d}:{:02d}".format(total_days, total_hours, total_minutes, total_seconds)

        t_exe_time_seconds = t_loop_time
        t_exe_days = t_exe_time_seconds// 86400
        t_exe_time_seconds %= 86400
        t_exe_hours = t_exe_time_seconds // 3600
        t_exe_time_seconds %= 3600
        t_exe_minutes = t_exe_time_seconds // 60
        t_exe_seconds = t_exe_time_seconds % 60
        total_exec_formatted_time = "{:02d}:{:02d}:{:02d}:{:02d}".format(int(t_exe_days), int(t_exe_hours), int(t_exe_minutes), int(t_exe_seconds))


        print('details - loop time - ', round(loop_time,2), 's   completed frames - ', frame_count,'/', total_frames,'   completed percentage - ', round((frame_count/total_frames)*100,2),'%   ast_finish_time - ', formatted_time,'   total_execution_time - ', total_exec_formatted_time, '   total_process_time - ', total_process_formatted_time)

        cv2.waitKey(1)

    video_capture.release()
    output_video.release()

if __name__ == '__main__':
    start_process("stu2 320x240.avi")
    # create_individual_videos()
    # predict_final_student_status()
    # write_to_csv()
    # read_csv()