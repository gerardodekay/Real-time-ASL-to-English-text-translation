import json
import os
import shutil
import moviepy.video.fx.all as vfx
from moviepy.editor import VideoFileClip
import re
import multiprocessing

VIDEOS_PATH = '/home/ubuntu/data/videos'
JSON_PATH = '/home/ubuntu/data'


def sanitize_file_name(filename):
    return re.sub('[^\w\-_\. ]', '_', filename)


def crop_video(split_json, split_name):
    for t in split_json:
        url = t["url"]
        bbox = t["box"]
        clean_text = t["clean_text"]
        crop_video_file(bbox, clean_text, split_name, url)


def crop_multi_process(split_json, split_name):
    vars = [(t["box"], t["clean_text"], split_name, t["url"]) for t in split_json]
    p = multiprocessing.Pool(multiprocessing.cpu_count()-1)
    p.starmap(crop_video_file, vars)


def crop_video_file(bbox, clean_text, split_name, url):
    file_name = url[url.index("v=") + 2:len(url)] + ".mp4"
    temp_file_name = sanitize_file_name("temp_" + file_name)
    split_file_dir = VIDEOS_PATH + "/" + split_name + "/" + clean_text
    split_file_path = split_file_dir + "/" + file_name
    temp_file_path = split_file_dir + "/" + temp_file_name
    if os.path.exists(split_file_path):
        clip = VideoFileClip(split_file_path)
        y1, x1, y2, x2 = bbox
        w = clip.w
        h = clip.h
        x1, x2 = x1 * w, x2 * w
        y1, y2 = y1 * h, y2 * h
        new_clip = clip.fx(vfx.crop, x1=x1, y1=y1, x2=x2, y2=y2)
        new_clip.write_videofile(temp_file_path)
        shutil.move(temp_file_path, split_file_path)
        print("Cropped %s %s %s" % (file_name, split_name, clean_text))


def crop_dataset():
    with open(JSON_PATH + "/MSASL_train.json") as f:
        train_json = json.load(f)
    with open(JSON_PATH + "/MSASL_test.json") as f:
        test_json = json.load(f)
    with open(JSON_PATH + "/MSASL_val.json") as f:
        val_json = json.load(f)

    crop_multi_process(train_json, "train")
    crop_multi_process(val_json, "val")
    crop_multi_process(test_json, "test")


if __name__ == "__main__":
    crop_dataset()
