from i3d import Inception3D
from pretrained_i3d import PreTrainedInception3d
import tensorflow as tf


def check_i3d():
    m = PreTrainedInception3d(include_top=True, input_shape=(40, 224, 224, 3))
    print(m.summary())

if __name__ == "__main__":
    check_i3d()