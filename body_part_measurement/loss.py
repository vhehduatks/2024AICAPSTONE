import numpy as np
import tensorflow as tf


def get_loss_fn(type_loss_fn = "wing") :
    if type_loss_fn == "wing" :
        return WingLoss
    elif type_loss_fn == "mae" :
        return tf.keras.losses.MeanAbsoluteError()
    else :
        return type_loss_fn

# Only available for denormalized data
def WingLoss(y_true, y_pred, w=10, epsilon=2):
    C = w - w * np.log(1 + w / epsilon)
    x = tf.math.abs(y_true - y_pred)
    x = tf.keras.backend.switch(x < w, w * tf.math.log(1 + x / epsilon), x - C)
    return tf.reduce_mean(x)
