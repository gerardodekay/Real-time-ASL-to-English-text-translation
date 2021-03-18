import json
import os
import shutil
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import moviepy.video.fx.all as vfx
from moviepy.editor import VideoFileClip
import re
import time

VIDEOS_PATH = '/home/ubuntu/data/videos'
JSON_PATH = '/home/ubuntu/data'
CROP_VIDEO_PATH = '/home/ubuntu/data/videos/crop'


def sanitize_file_name(filename):
    return re.sub('[^\w\-_\. ]', '_', filename)

def crop_video_file(trimmed_file, bbox):
    try:
        file_name = trimmed_file.split("/")[-1]
        class_name = trimmed_file.split("/")[-2]
        split_name = trimmed_file.split("/")[-3]
        temp_file_name = sanitize_file_name("temp_" + file_name)
        crop_file_dir = os.path.join(CROP_VIDEO_PATH, split_name, class_name)
        if not os.path.exists(crop_file_dir):
            os.makedirs(crop_file_dir)
        temp_file_path = os.path.join(crop_file_dir, temp_file_name)
        crop_file_path = os.path.join(crop_file_dir, file_name)

        if os.path.exists(trimmed_file) and not os.path.exists(crop_file_path):
            clip = VideoFileClip(trimmed_file)
            y1, x1, y2, x2 = bbox
            w = clip.w
            h = clip.h
            x1, x2 = x1 * w, x2 * w
            y1, y2 = y1 * h, y2 * h
            new_clip = clip.fx(vfx.crop, x1=x1, y1=y1, x2=x2, y2=y2)
            new_clip.write_videofile(temp_file_path)
            time.sleep(2)
            shutil.move(temp_file_path, crop_file_path)
            print("Cropped %s %s %s" % (file_name, split_name, class_name))
    except Exception as ex:
        print('Error Cropping')
        print(ex)


def copy_split(split_json, split_name="train"):
    split_classes = []
    split_misses = []
    count = 0
    if not os.path.exists(VIDEOS_PATH + "/" + split_name):
        os.mkdir(VIDEOS_PATH + "/" + split_name)
    for t in split_json:
        url = t["url"]
        start_time = t["start_time"]
        end_time = t["end_time"]
        bbox = t["box"]
        file_name = url[url.index("v=")+2:len(url)] + ".mp4"
        file_path = VIDEOS_PATH + "/" + file_name
        
        target_dir = VIDEOS_PATH + "/" + split_name + "/" + t["clean_text"]
        target_path = target_dir + "/" + file_name
        
        TrimmedVideo_Path = VIDEOS_PATH+"/TrimmedVideos/"
        TrimmedVideo_TargetPath = TrimmedVideo_Path+file_name
                
        if not os.path.exists(TrimmedVideo_Path):
                os.mkdir(TrimmedVideo_Path)
                #print("Making TrimmedVideo_Path dir",TrimmedVideo_Path)
        if os.path.exists(file_path):
            #print("Found the file to trim",file_path)
            ffmpeg_extract_subclip(file_path, start_time, end_time, targetname=TrimmedVideo_TargetPath)    
            #print("Trimming done")
        if os.path.exists(TrimmedVideo_TargetPath):
            if not os.path.exists(target_dir):
                os.mkdir(target_dir)
            if not os.path.exists(target_path):
                split_classes.append(t["clean_text"])
                shutil.move(TrimmedVideo_TargetPath, target_path)
                crop_video_file(target_path, bbox)
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


if __name__ == "__main__":
    split_data()
