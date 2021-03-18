import sys 

# import cv2
import os
import pickle
from os.path import join, exists
# import segment_hand as hs
import json 
import pandas as pd
import numpy as np
from random import shuffle
from math import floor
from os import listdir
from os.path import isfile, join
import os
import shutil
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


#read json file containing video url with other details required for preprocessing like start time, end time, etc.
MSASL_trainData = pd.read_json('data/MSASL_train.json')
MSASL_valData = pd.read_json('data/MSASL_val.json')
MSASL_testData = pd.read_json('data/MSASL_test.json')
MSASL_classes = pd.read_json('data/MSASL_classes.json')
MSASL_classes.columns = ['class']

MSASL_Data = pd.concat([MSASL_trainData, MSASL_valData, MSASL_testData], ignore_index=True)

# new data frame with url splitted to get the video name
split_df = MSASL_Data["url"].str.split("=", n = 1, expand = True) 
  
# making separate Video name column from new data frame 
MSASL_Data["VideoName"]= split_df[1]

def TrimVideoClip(data_dir):
    video_dir = os.getcwd() + data_dir + "/"
    home_directory = os.getcwd() + data_dir
    files = [f for f in listdir(home_directory) if isfile(join(home_directory, f))]

    for file_name in files:
        fileName = (file_name[:-4])
        VideoNameDF = MSASL_Data.loc[MSASL_Data['VideoName'] == fileName] #Filter for the file name in the df
        if VideoNameDF.empty:
            continue
        start_time = VideoNameDF['start_time'].min() # read the corresponding start and end time for the video from the df; min(), max() are just a proxy; we expect start and end time to be same for a given video name in case multiple enteries are present for the video
        end_time = VideoNameDF['end_time'].max()
        print(fileName,start_time, end_time)
        videoInput_path = video_dir + file_name
        TrimmedVideo_TargetPath = video_dir + "/TrimmedVideos/"
        
        if not os.path.exists(TrimmedVideo_TargetPath):
                os.mkdir(TrimmedVideo_TargetPath)
        
        ffmpeg_extract_subclip(videoInput_path, start_time, end_time, targetname=TrimmedVideo_TargetPath+file_name)

def copy_split(split_json, split_name="train"):
    split_classes = []
    split_misses = []
    if not os.path.exists(VIDEOS_PATH + "/" + split_name):
        os.mkdir(VIDEOS_PATH + "/" + split_name)
    for t in split_json:
        url = t["url"]
        file_name = url[url.index("v=")+2:len(url)] + ".mp4"
        file_path = VIDEOS_PATH + "/videos/" + file_name
        target_dir = VIDEOS_PATH + "/" + split_name + "/" + t["clean_text"]
        target_path = target_dir + "/" + file_name
        if os.path.exists(file_path):
            if not os.path.exists(target_dir):
                os.mkdir(target_dir)
            if not os.path.exists(target_path):
                split_classes.append(t["clean_text"])
                shutil.copy(file_path, target_path)
        else:
            split_misses.append((file_name, url))
    return split_classes, split_misses


def split_data():
    train_f = JSON_PATH + "/MSASL_train.json"
    classes_f = JSON_PATH + "/MSASL_classes.json"
    test_f = JSON_PATH + "/MSASL_test.json"
    val_f = JSON_PATH + "/MSASL_val.json"

    with open(train_f) as f:
        train_json = json.load(f)
    with open(classes_f) as f:
        classes_json = json.load(f)
    with open(test_f) as f:
        test_json = json.load(f)
    with open(val_f) as f:
        val_json = json.load(f)

    train_classes, train_misses = copy_split(train_json, "train")
    val_classes, val_misses = copy_split(val_json, "val")
    test_classes, test_misses = copy_split(test_json, "test")

    print("test")



def convert_to_frames(Inputdata_path,word_count,input_type):
    """
    Takes Raw training Input dataset and converts them from video to frames 
    """
    
    # Create folder to store frames for all words 
    rootPath = os.getcwd()
    # need to change image data for different conversions 
    image_data = os.path.join(os.getcwd(), "MSData/image_data_" + input_type)
    if (not exists(image_data)):
        os.makedirs(image_data)
        
    frame_count = 0
    Inputdata_path = os.path.join(os.getcwd(), Inputdata_path, input_type)
    os.chdir(Inputdata_path)
   # print("Inputdata_path: ",Inputdata_path)
    
    # Get all files with raw data for words, only keep how many you want
    gesture_list = os.listdir(os.getcwd())
    #gesture_list = gesture_list[:word_count]
    
    for gesture in gesture_list:
        gesture_path = os.path.join(Inputdata_path, gesture)
        os.chdir(gesture_path)
        #print("gesture_path: ",gesture_path)
        
        # Create directory to store images
        frames = os.path.join(image_data, gesture)
        if(not os.path.exists(frames)):
            os.makedirs(frames)
        #print("frames", frames)
        
        videos = mylistdir(os.getcwd())
        videos = [video for video in videos if(os.path.isfile(os.getcwd() + '/' +  video))]

        for video in videos:
            video_name = video[:-4] #removing .mp4 from the video name
            #print("video_name: ", video_name)
            vidcap = cv2.VideoCapture(video)
            success,image = vidcap.read()
            frame_count = 0
            os.chdir(frames)
            while success:
              # image = cv2.cvtcolor(image,cv2.color_bgr2gray) # to convert image to grayscale
              cv2.imwrite("%s_frame%d.jpg" % (video_name,frame_count), image)     # save frame as jpeg file      
              success,image = vidcap.read()
              print('read a new frame: ', success)
              frame_count += 1
    
if __name__ == '__main__':

    JSON_PATH = '/home/ubuntu/data'
    VIDEOS_PATH = '/data/videos'
    TrimmedVideos_PATH = '/home/ubuntu/data/videos/TrimmedVideos'

    TrimVideoClip(VIDEOS_PATH)
    # split_data(TrimmedVideos_PATH,JSON_PATH)

    # convert_to_frames("MSData/",10,"train")
    # convert_to_frames("MSData/",10,"test")