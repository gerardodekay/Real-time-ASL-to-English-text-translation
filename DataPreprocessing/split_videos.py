import json
import os
import shutil

VIDEOS_PATH = '/home/anilsrik/anil/cs230/data'


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
    train_f = VIDEOS_PATH + "/MSASL_train.json"
    classes_f = VIDEOS_PATH + "/MSASL_classes.json"
    test_f = VIDEOS_PATH + "/MSASL_test.json"
    val_f = VIDEOS_PATH + "/MSASL_val.json"

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
