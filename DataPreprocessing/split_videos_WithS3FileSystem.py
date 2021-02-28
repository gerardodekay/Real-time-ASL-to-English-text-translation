import json
import os
import shutil
import boto3
import pandas as pd
from sagemaker import get_execution_role
from PIL import Image
import s3fs

fs = s3fs.S3FileSystem()

role = get_execution_role()
bucket='cs230projectdata'
data_key = 'TrimmedVideos'
VIDEOS_PATH = 's3://{}/{}'.format(bucket, data_key)
JSON_PATH = '/home/ec2-user/SageMaker/Real-time-ASL-to-English-text-translation/data/msasl'

def copy_split(split_json, split_name="train"):
    split_classes = []
    split_misses = []
    if not fs.exists(VIDEOS_PATH + "/" + split_name):
        fs.mkdir(VIDEOS_PATH + "/" + split_name)
    for t in split_json:
        url = t["url"]
        file_name = url[url.index("v=")+2:len(url)] + ".mp4"
        file_path = VIDEOS_PATH + "/" + file_name
        target_dir = VIDEOS_PATH + "/" + split_name + "/" + t["clean_text"]
        target_path = target_dir + "/" + file_name
        if fs.exists(file_path):
            if not fs.exists(target_dir):
                fs.mkdir(target_dir)
            if not fs.exists(target_path):
                split_classes.append(t["clean_text"])
                fs.copy(file_path, target_path)
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


if __name__ == "__main__":
    split_data()
