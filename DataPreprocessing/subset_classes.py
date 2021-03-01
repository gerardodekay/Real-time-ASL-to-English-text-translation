import os

PATH = '/home/anilsrik/anil/cs230/data'


def get_classes(count=100):
    train_path = PATH + '/train'
    test_path = PATH + '/test'
    val_path = PATH + '/val'
    train_classes = os.listdir(train_path)
    train_cls_sz = []
    test_cls_sz = []
    val_cls_sz = []
    for cl in train_classes:
        train_cls_sz.append(sum(os.path.getsize(train_path + '/' + cl + '/' + f) for f in os.listdir(train_path + '/' + cl)))
    test_classes = os.listdir(test_path)
    for cl in test_classes:
        test_cls_sz.append(sum(os.path.getsize(test_path + '/' + cl + '/' + f) for f in os.listdir(test_path + '/' + cl)))

    val_classes = os.listdir(val_path)
    for cl in val_classes:
        val_cls_sz.append(sum(os.path.getsize(val_path + '/' + cl + '/' + f) for f in os.listdir(val_path + '/' + cl)))
    # Sort by each class folder size
    train_dict = {k: v for k, v in sorted(dict(zip(train_classes, train_cls_sz)).items(), key=lambda item: item[1])}
    test_dict = {k: v for k, v in sorted(dict(zip(test_classes, test_cls_sz)).items(), key=lambda item: item[1])}
    val_dict = {k: v for k, v in sorted(dict(zip(val_classes, val_cls_sz)).items(), key=lambda item: item[1])}

    train_classes = list(train_dict.keys())
    test_classes = list(test_dict.keys())
    val_classes = list(val_dict.keys())

    common_classes = list(set(train_classes).intersection(set(test_classes)).intersection(set(val_classes)))
    return common_classes[:count]


if __name__=='__main__':
    print(get_classes())