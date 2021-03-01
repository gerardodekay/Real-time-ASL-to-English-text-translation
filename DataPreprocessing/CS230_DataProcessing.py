import sys 

import cv2
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
MSASL_trainData = pd.read_json(r'MSASL_train.json')
MSASL_valData = pd.read_json(r'MSASL_val.json')
MSASL_testData = pd.read_json(r'MSASL_test.json')
MSASL_classes = pd.read_json(r'MSASL_classes.json')
MSASL_classes.columns = ['class']

MSASL_Data = pd.concat([MSASL_trainData, MSASL_valData, MSASL_testData], ignore_index=True)

# new data frame with url splitted to get the video name
split_df = MSASL_Data["url"].str.split("=", n = 1, expand = True) 
  
# making separate last name column from new data frame 
MSASL_Data["VideoName"]= split_df[1]

def TrimVideoClip(data_dir):
    # data_dir = "/MSData/subset"
    video_dir = os.getcwd() + data_dir + "/"
    home_directory = os.getcwd() + data_dir
    files = [f for f in listdir(home_directory) if isfile(join(home_directory, f))]

    for file_name in files:
        fileName = (file_name[:-4])
        VideoNameDF = MSASL_Data.loc[MSASL_Data['VideoName'] == fileName] #Filter for the file name in the df
        if VideoNameDF.empty:
            continue
        start_time = VideoNameDF['start_time'].min() # read the corresponding start and end time for the vide from the df; min(), max() are just a proxy; we expect start and end time to be same for a given video name in case multiple enteries are present for the video
        end_time = VideoNameDF['end_time'].max()
        print(fileName,start_time, end_time)
        print(end_time)
        videoInput_path = video_dir + file_name
        TrimmedVideo_TargetPath = video_dir + "/TrimmedVideos/"
        
        if not os.path.exists(TrimmedVideo_TargetPath):
                os.mkdir(TrimmedVideo_TargetPath)
        
        ffmpeg_extract_subclip(videoInput_path, start_time, end_time, targetname=TrimmedVideo_TargetPath+file_name)


def sort_files(data_dir):

    data_dir = "/MSData/subset/TrimmedVideos"
    home_directory = os.getcwd() + data_dir
    files = [f for f in listdir(home_directory) if isfile(join(home_directory, f))]

    df = MSASL_Data[['clean_text','VideoName']]

    name_dict = df.groupby(['clean_text'])['VideoName'].apply(list).to_dict()

    for className, Videos in name_dict.items():
        for videoName in Videos:
            output_names = [f for f in files if (f[:-4] == videoName)]
            for file_name in output_names: 
                if not os.path.exists(home_directory + "/" +  className):
                    os.mkdir(home_directory + "/" + className)
                current_directory = home_directory + "/" + file_name
                print(current_directory)
                new_directory = home_directory + "/" +  className + "/" + file_name
                print('new_directory',new_directory)
                if os.path.exists(current_directory):
                    shutil.move(current_directory, new_directory)
                    print(className, " Moved!")

def mylistdir(directory):
    """A specialized version of os.listdir() that ignores files that
    start with a leading period."""
    filelist = os.listdir(directory)
    return [x for x in filelist
            if not (x.startswith('.'))]

def split_test_train(main_dir,data_folder_path):
    """
    Moves raw video data into training (70%) and testing (30%) sets 
    """
    main_dir = os.getcwd() + main_dir
    data_dir = main_dir + data_folder_path

    all_files = mylistdir(os.path.abspath(data_dir))
    for file_name in all_files:
        # Get a list of the files
        file_direc = os.path.abspath(data_dir + file_name)
        if(os.path.isdir(file_direc)):
            data_files = list(filter(lambda file: file.endswith('mp4'), mylistdir(file_direc)))

            # Randomize the files
            shuffle(data_files)

            #Split files into training and testing sets
            split = 0.7
            split_index = floor(len(data_files) * split)
            training = data_files[:split_index]
            testing = data_files[split_index:]

            train_dir = main_dir + "train/"
            test_dir = main_dir + "test/"

            if(not os.path.exists(train_dir)):
                os.makedirs(train_dir)

            if(not os.path.exists(test_dir)):
                os.makedirs(test_dir)
	    
            for file in training:
                from_dir = data_dir + file_name + "/" + file
                to_dir =  train_dir + file_name 
                if(not os.path.exists(to_dir)):
                    os.makedirs(to_dir)

                to_dir += "/" + file
                shutil.move(from_dir, to_dir)

            for file in testing:
                from_dir = data_dir + file_name + "/" + file
                to_dir =  test_dir + file_name 
                if(not os.path.exists(to_dir)):
                    os.makedirs(to_dir)

                to_dir += "/" + file
                shutil.move(from_dir, to_dir)

            os.rmdir(file_direc)
            print("Done Splitting Dataset")

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
    
    data_dir = "/MSData/subset"
    sort_dir = data_dir + "/TrimmedVideos"

    TrimVideoClip(data_dir)
    
    # sort_files(sort_dir)

    # split_dir = "/MSData/"
    # data_folder_path = "subset/TrimmedVideos/"
    # split_test_train(split_dir, data_folder_path)
    # convert_to_frames("MSData/",10,"train")
    # convert_to_frames("MSData/",10,"test")
    # def convert_to_frames(dataset,word_count,input_type,output_pickle_name)