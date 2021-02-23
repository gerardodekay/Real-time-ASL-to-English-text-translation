from i3d import Inception3D
import tensorflow as tf

def check_i3d():
    m = Inception3D()
    m.build(input_shape=(None, 64, 224, 224, 3))
    m.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])

    print(m.summary())


if __name__ == "__main__":
    check_i3d()