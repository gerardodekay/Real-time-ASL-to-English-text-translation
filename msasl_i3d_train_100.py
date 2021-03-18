from pretrained_i3d import PreTrainedInception3d, layers_freeze, add_top_layer
# from i3d import Inception3D
import tensorflow as tf
import keras_video
import keras_video.utils
import os
import time
import json
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator


train_glob_pattern = '/home/ubuntu/data/videos/crop/train/{classname}/*.mp4'
val_glob_pattern = '/home/ubuntu/data/videos/crop/val/{classname}/*.mp4'


# def check_i3d():
#     m_rgb = PreTrainedInception3d(include_top=True, pretrained_weights="rgb_imagenet_and_kinetics", input_shape=(80, 224, 224, 3))
#     #m_flow = PreTrainedInception3d(include_top=True, pretrained_weights="flow_imagenet_and_kinetics", input_shape=(40, 224, 224, 2))
#     #print(m_flow.summary())
#     print(m_rgb.summary())


def model_checkpoints():
    model_dir = "model"
    checkpoint = time.strftime("%Y%m%d-%H%M", time.gmtime()) + "-%s%03d-oflow-i3d" % ("ASL", 105)
    os.makedirs(model_dir, exist_ok=True)
    cpTopLast = tf.keras.callbacks.ModelCheckpoint(filepath=model_dir + "/" + checkpoint + "-top-last.h5", verbose=1,
                                                save_best_only=False, save_weights_only=False)
    cpTopBest = tf.keras.callbacks.ModelCheckpoint(filepath=model_dir + "/" + checkpoint + "-top-best.h5", verbose=1,
                                                save_best_only=True, save_weights_only=False)
    cbTensorBoard = tf.keras.callbacks.TensorBoard(log_dir="logs", histogram_freq=1, update_freq='batch',
                                                write_graph=True, write_images=True)
    return [cpTopLast, cpTopBest, cbTensorBoard]


def run_i3d_pretrained():
    EPOCHS = 50
    with open("/home/ubuntu/data/MSASL_classes.json") as f:
        classes = json.load(f)

    classes = ['bath', 'calculus', 'israel', 'tall', 'woman', 'hospital', 'sleep', 'knife', 'hungry', 'snowboard',
               'visit', 'sunday', 'use', 'engaged', 'ocean', 'bat', 'pineapple', 'water', '24', 'niece', 'good',
               'how_many', 'flower', 'car', 'appointment', 'pay', 'tie', 'accept', 'cabinets', 'fourteen', 'museum',
               'engineering', 'cheap', 'artist', 'never', 'dizzy', 'eyes', 'blanket', 'his', 'me', 'bear', 'drive',
               'store', 'new', 'football', 'lost', 'bitter', 'sandwich', 'teacher', 'refrigerator', 'only', 'four',
               'walk', 'radio', 'philippines', 'curly hair', 'restaurant', 'become', 'therapy', 'should', 'university',
               'bad', 'skateboard', 'bird', 'next week', 'put_down', 'destroy', 'glue', 'hill', 'fast', 'stop',
               'toast', 'couch', 'hard', '27', 'not understand', 'every morning', 'swing', 'all night', 'come',
               'they', 'baseball', 'wrong', 'son', 'salt', 'wow', 'motorcycle', 'curtains', 'seven', 'skiing',
               'dark', 'screwdriver', 'disappear', 'flat tire', 'police', 'famous', 'equal', 'apply', 'thank you',
               'girlfriend']
    train_datagen = ImageDataGenerator(width_shift_range=0.2,height_shift_range=0.2)
    train = keras_video.VideoFrameGenerator(classes=classes, nb_frames=64, batch_size=10, use_headers=False, transformation=train_datagen, glob_pattern=train_glob_pattern)
    val = keras_video.VideoFrameGenerator(classes=classes, nb_frames=64, batch_size=10, use_headers=False,  glob_pattern=val_glob_pattern)
    m_rgb = PreTrainedInception3d(include_top=False, pretrained_weights="rgb_imagenet_and_kinetics", dropout_prob=0.5,
                                  input_shape=(64, 224, 224, 3), classes=100)

    m_rgb = layers_freeze(m_rgb)
    print("Freezing layers done")
    m_rgb = add_top_layer(m_rgb, classes=100, dropout_prob=0.5)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, decay=1e-6)
    m_rgb.compile(optimizer, 'categorical_crossentropy', metrics=['accuracy'])
    model_ckpts = model_checkpoints()
    m_rgb.fit_generator(
        train,
        validation_data=val,
        verbose=1,
        epochs=EPOCHS,
        callbacks=model_ckpts
    )

if __name__ == "__main__":
    run_i3d_pretrained()
