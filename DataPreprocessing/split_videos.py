import json
import os
import shutil
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

VIDEOS_PATH = '/home/ubuntu/data/videos'
JSON_PATH = '/home/ubuntu/data'

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
