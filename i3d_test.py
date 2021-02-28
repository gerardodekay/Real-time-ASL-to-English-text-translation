from i3d import Inception3D
from pretrained_i3d import PreTrainedInception3d, layers_freeze, add_top_layer
import tensorflow as tf
import keras_video
import keras_video.utils

train_glob_pattern = '/home/anilsrik/anil/cs230/data/train/{classname}/*.mp4'
val_glob_pattern = '/home/anilsrik/anil/cs230/data/val/{classname}/*.mp4'


# def check_i3d():
#     m_rgb = PreTrainedInception3d(include_top=True, pretrained_weights="rgb_imagenet_and_kinetics", input_shape=(80, 224, 224, 3))
#     #m_flow = PreTrainedInception3d(include_top=True, pretrained_weights="flow_imagenet_and_kinetics", input_shape=(40, 224, 224, 2))
#     #print(m_flow.summary())
#     print(m_rgb.summary())


def run_i3d_pretrained():
    EPOCHS = 2
    classes = ["about", "accept", "absent", "acquire"]
    train = keras_video.VideoFrameGenerator(classes=classes, nb_frames=24, batch_size=2, glob_pattern=train_glob_pattern)
    val = keras_video.VideoFrameGenerator(classes=classes, nb_frames=24, batch_size=2, glob_pattern=val_glob_pattern)
    m_rgb = PreTrainedInception3d(include_top=False, pretrained_weights="rgb_imagenet_and_kinetics", dropout_prob=0.5,
                                  input_shape=(24, 224, 224, 3), classes=4)

    m_rgb = layers_freeze(m_rgb)
    print("Freezing layers done")
    m_rgb = add_top_layer(m_rgb, classes=4, dropout_prob=0.5)
    optimizer = tf.keras.optimizers.Adam(0.001)
    m_rgb.compile(optimizer, 'categorical_crossentropy', metrics=['accuracy'])
    m_rgb.fit_generator(
        train,
        validation_data=val,
        verbose=1,
        epochs=EPOCHS
    )


if __name__ == "__main__":
    run_i3d_pretrained()