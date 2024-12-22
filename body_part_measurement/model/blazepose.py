import tensorflow as tf
from tensorflow.keras.models import Model
from .blazepose_layers import BlazeBlock

class BlazePose():
    def __init__(self, batch_size, input_shape, num_keypoints, num_seg_channels, attention_model=None):
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.attention_model = attention_model
        self.num_keypoints = num_keypoints
        self.num_seg_channels = num_seg_channels

        self.conv1 = tf.keras.layers.Conv2D(
            filters=24, kernel_size=3, strides=(2, 2), padding='same', activation='relu'
        )

        self.conv2_1 = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=3, padding='same', activation=None),
            tf.keras.layers.Conv2D(filters=24, kernel_size=1, activation=None)
        ])

        self.conv2_2 = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=3, padding='same', activation=None),
            tf.keras.layers.Conv2D(filters=24, kernel_size=1, activation=None)
        ])

        # === Heatmap ===

        self.conv3 = BlazeBlock(block_num=3, channel=48)
        self.conv4 = BlazeBlock(block_num=4, channel=96)
        self.conv5 = BlazeBlock(block_num=5, channel=192)
        self.conv6 = BlazeBlock(block_num=6, channel=288)

        self.conv7a = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=3, padding="same", activation=None),
            tf.keras.layers.Conv2D(
                filters=48, kernel_size=1, activation="relu"),
            tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="bilinear")
        ])
        self.conv7b = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=3, padding="same", activation=None),
            tf.keras.layers.Conv2D(
                filters=48, kernel_size=1, activation="relu")
        ])

        self.conv8a = tf.keras.layers.UpSampling2D(
            size=(2, 2), interpolation="bilinear")
        self.conv8b = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=3, padding="same", activation=None),
            tf.keras.layers.Conv2D(
                filters=48, kernel_size=1, activation="relu")
        ])

        self.conv9a = tf.keras.layers.UpSampling2D(
            size=(2, 2), interpolation="bilinear")
        self.conv9b = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=3, padding="same", activation=None),
            tf.keras.layers.Conv2D(
                filters=48, kernel_size=1, activation="relu")
        ])

        self.conv10a = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=3, padding="same", activation=None),
            tf.keras.layers.Conv2D(
                filters=8, kernel_size=1, activation="relu"),
            tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="bilinear")
        ])
        self.conv10b = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=3, padding="same", activation=None),
            tf.keras.layers.Conv2D(filters=8, kernel_size=1, activation="relu")
        ])

        self.conv11 = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=3, padding="same", activation=None),
            tf.keras.layers.Conv2D(
                filters=8, kernel_size=1, activation="relu"),
            tf.keras.layers.Conv2D(
                filters=self.num_seg_channels, kernel_size=3, padding="same", activation=None) # -> Heatmap output
        ])

        # === Regression ===

        #  In: 1, 64, 64, 48)
        self.conv12a = BlazeBlock(block_num=4, channel=96, name_prefix="regression_conv12a_")    # input res: 64
        self.conv12b = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=3, padding="same", activation=None, name="regression_conv12b_depthwise"),
            tf.keras.layers.Conv2D(
                filters=96, kernel_size=1, activation="relu", name="regression_conv12b_conv1x1")
        ], name="regression_conv12b")

        self.conv13a = BlazeBlock(block_num=5, channel=192, name_prefix="regression_conv13a_")   # input res: 32
        self.conv13b = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=3, padding="same", activation=None, name="regression_conv13b_depthwise"),
            tf.keras.layers.Conv2D(
                filters=192, kernel_size=1, activation="relu", name="regression_conv13b_conv1x1")
        ], name="regression_conv13b")

        if attention_model is not None :
            self.conv14a = BlazeBlock(block_num=6, channel=289, name_prefix="regression_conv14a_")   # input res: 16
            self.conv14b = tf.keras.models.Sequential([
                tf.keras.layers.DepthwiseConv2D(
                    kernel_size=3, padding="same", activation=None, name="regression_conv14b_depthwise"),
                tf.keras.layers.Conv2D(
                    filters=289, kernel_size=1, activation="relu", name="regression_conv14b_conv1x1")
            ], name="regression_conv14b")
        else :
            self.conv14a = BlazeBlock(block_num=6, channel=288, name_prefix="regression_conv14a_")   # input res: 16
            self.conv14b = tf.keras.models.Sequential([
                tf.keras.layers.DepthwiseConv2D(
                    kernel_size=3, padding="same", activation=None, name="regression_conv14b_depthwise"),
                tf.keras.layers.Conv2D(
                    filters=288, kernel_size=1, activation="relu", name="regression_conv14b_conv1x1")
            ], name="regression_conv14b")

        if attention_model is not None :
            self.conv15 = tf.keras.models.Sequential([
                BlazeBlock(block_num=7, channel=290, channel_padding=0, name_prefix="regression_conv15a_"),
                BlazeBlock(block_num=7, channel=290, channel_padding=0, name_prefix="regression_conv15b_")
            ], name="regression_conv15")
        else :
            self.conv15 = tf.keras.models.Sequential([
                BlazeBlock(block_num=7, channel=288, channel_padding=0, name_prefix="regression_conv15a_"),
                BlazeBlock(block_num=7, channel=288, channel_padding=0, name_prefix="regression_conv15b_")
            ], name="regression_conv15")

        self.conv16 = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(
                filters=self.num_keypoints, kernel_size=2, activation=None)
        ], name="joints")

    def build_model(self, model_type):

        input_x = tf.keras.layers.Input(shape=(self.input_shape[0], self.input_shape[1], 3), batch_size=self.batch_size)

        # Block 1
        # In: 1x256x256x3
        x = self.conv1(input_x)

        # Block 2
        # In: 1x128x128x24
        x = x + self.conv2_1(x)
        x = tf.keras.activations.relu(x)

        # Block 3
        # In: 1x128x128x24
        x = x + self.conv2_2(x)
        y0 = tf.keras.activations.relu(x)

        # === Heatmap ===

        # In: 1, 128, 128, 24
        y1 = self.conv3(y0)
        y2 = self.conv4(y1)
        y3 = self.conv5(y2)
        y4 = self.conv6(y3)

        x = self.conv7a(y4) + self.conv7b(y3)
        x = self.conv8a(x) + self.conv8b(y2)
        # In: 1, 32, 32, 96
        x = self.conv9a(x) + self.conv9b(y1)
        # In: 1, 64, 64, 48
        y = self.conv10a(x) + self.conv10b(y0)
        y = self.conv11(y)

        # In: 1, 128, 128, 8
        segs = y

        # === Regression ===

        # Stop gradient for regression on 2-head model
        if model_type == "REGRESSION_AND_SEGMENTATION":
            x = tf.keras.backend.stop_gradient(x)
            y2 = tf.keras.backend.stop_gradient(y2)
            y3 = tf.keras.backend.stop_gradient(y3)
            y4 = tf.keras.backend.stop_gradient(y4)

        x = self.conv12a(x) + self.conv12b(y2)
        if self.attention_model is not None :
            x = tf.concat([x, self.attention_model.outputs[0]], axis=3)
        # In: 1, 32, 32, 96
        x = self.conv13a(x) + self.conv13b(y3)
        if self.attention_model is not None :
            x = tf.concat([x, self.attention_model.outputs[1]], axis=3)
        # In: 1, 16, 16, 192
        x = self.conv14a(x) + self.conv14b(y4)
        if self.attention_model is not None :
            x = tf.concat([x, self.attention_model.outputs[2]], axis=3)
        # In: 1, 8, 8, 288
        x = self.conv15(x)
        # In: 1, 2, 2, 288
        joints = self.conv16(x)
        output = tf.reshape(joints, (-1,self.num_keypoints))

        if self.attention_model is not None :
            inputs = [input_x, self.attention_model.input]
        else :
            inputs = input_x

        if model_type == "REGRESSION_AND_SEGMENTATION":
            return Model(inputs=inputs, outputs=[output, segs])
        elif model_type == "SEGMENTATION":
            return Model(inputs=inputs, outputs=segs)
        elif model_type == "REGRESSION":
            return Model(inputs=inputs, outputs=output)
        else:
            raise ValueError("Wrong model type.")
            