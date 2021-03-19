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

    classes = ['match', 'fail', 'laugh', 'book', 'sign language', 'school', 'easter', 'boring', 'past', 'phone', 'library', 'germany',
              'like', 'cochlear implant', 'rainbow', 'letter', 'from', 'his', 'hong kong', 'every morning', 'france', 'give', 'shock', 'puerto rico',
              'portugal', 'breakfast', 'day', 'afternoon', 'happy', 'sad', 'nervous', 'upset', 'want', 'know', 'must', 'how', 'wow', 'pass', 'think',
              'yellow', 'teacher', 'deaf', 'university', 'fix', 'clock', 'there', 'next week', 'again', 'sick', 'meat', 'important', 'football', 'mother',
              'father', 'milk', 'eat', 'drink', 'hot', 'cold', 'more', 'help', 'clothes', 'you', 'your', 'down', 'hurt', 'not like', 'mad', 'funny', 'silly', 
              'hungry', 'full', 'tired', 'where', 'dirty', 'play', 'outside', 'home', 'sister', 'brother', 'grandmother', 'grandfather', 'aunt', 'uncle', 
              'cousin', 'i love you', 'warm', 'stand', 'wait', 'business', 'blue', 'can', 'throw', 'fire', 'any', 'win', 'socks', 'gone', 'rude', 'bake',
              'wine', 'norway', 'hearing', 'spain', 'elementary school', 'center school', 'high school', 'gallaudet', 'english', 'spanish', 'not', 'oh i see', 
              'remember', 'forget', 'ball', '25', 'in', 'fish', 'fly', 'bed', 'marry', 'enjoy', 'meet', 'brown', 'spider', 'cafeteria', 'jacket', 'cereal',
              'tiger', 'south', 'dining room', 'banana', 'hit', 'work', 'actor', 'live', 'america', 'beach', 'birthday', 'black', 'celebrate', 'christmas', 
              'city', 'color', 'cool', 'dark', 'email', 'autumn', 'far', 'gray', 'green', 'spring up', 'halloween', 'hanukkah', 'here', 'how_many', 'inside',
              'internet', 'light', 'listen', 'march', 'month', 'music', 'musician', 'my', 'new', 'number', 'ocean', 'old', 'orange', 'our', 'pink', 'purple', 
              'rain', 'red', 'street', 'ski', 'snow', 'spring', 'summer', 'television', 'thanksgiving', 'their', 'vacation', 'visit', 'watch', 'weather', 'white',
              'winter', 'year', 'love it', 'money', 'shirt', 'curly hair', 'motorcycle', 'japan', 'same', 'cat', 'dance', 'turkey']   
    train_datagen = ImageDataGenerator(width_shift_range=0.2,height_shift_range=0.2)
    train = keras_video.VideoFrameGenerator(classes=classes, nb_frames=12, batch_size=5, transformation=train_datagen, glob_pattern=train_glob_pattern)
    val = keras_video.VideoFrameGenerator(classes=classes, nb_frames=12, batch_size=5, glob_pattern=val_glob_pattern)
    m_rgb = PreTrainedInception3d(include_top=False, pretrained_weights="rgb_imagenet_and_kinetics", dropout_prob=0.5,
                                  input_shape=(12, 224, 224, 3), classes=200)

    m_rgb = layers_freeze(m_rgb)
    print("Freezing layers done")
    m_rgb = add_top_layer(m_rgb, classes=200, dropout_prob=0.5)
    optimizer = tf.keras.optimizers.Adam(lr=0.0001, decay=1e-6)
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
