import keras
import numpy as np
from keras import layers
import tensorflow as tf
from keras.src.optimizers import Adam
import pathlib


class Model:
    def __init__(self, model_name="default", verbose=1):
        current_dir = pathlib.Path(__file__).parent.parent
        model_dir = current_dir / "resources" / "models" / model_name
        self.weights_path = model_dir / "model_data.weights.h5"
        self.model_name = model_name
        self.unet = None

    def double_conv_block(self, x, n_filters):
        # Conv2D then ReLU activation
        x = layers.Conv2D(n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)
        # Conv2D then ReLU activation
        x = layers.Conv2D(n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)
        return x

    # Downsample Block
    def downsample_block(self, x, n_filters):
        f = self.double_conv_block(x, n_filters)
        p = layers.MaxPool2D(2)(f)
        p = layers.Dropout(0.3)(p)
        return f, p

    def upsample_block(self, x, conv_features, n_filters):

        upsampling = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
        concatenation = layers.concatenate([upsampling, conv_features])
        dropout = layers.Dropout(0.3)(concatenation)

        # Conv2D twice with ReLU activation
        full_block = self.double_conv_block(dropout, n_filters)
        return full_block

    def build_unet_model(self):
        inputs = layers.Input(shape=(320, 320, 1))

        f1, p1 = self.downsample_block(inputs, 64)
        f2, p2 = self.downsample_block(p1, 128)
        f3, p3 = self.downsample_block(p2, 256)
        f4, p4 = self.downsample_block(p3, 512)
        bottleneck = self.double_conv_block(p4, 1024)
        u6 = self.upsample_block(bottleneck, f4, 512)
        u7 = self.upsample_block(u6, f3, 256)
        u8 = self.upsample_block(u7, f2, 128)
        u9 = self.upsample_block(u8, f1, 64)

        outputs = layers.Conv2D(1, 1, padding="same", activation="sigmoid")(u9)

        model = keras.Model(inputs, outputs, name="U-Net")

        return model

    def input_tensor_preprocessing(self, image):
        x_image = image
        if x_image.ndim == 2:
            x_image = np.expand_dims(x_image, axis=0)
        x_image = np.expand_dims(x_image, axis=-1)
        x_image = tf.convert_to_tensor(x_image)
        x_image = x_image / np.max(x_image)
        x_image = tf.cast(x_image, dtype=tf.float64)
        return x_image

    def load_model(self):
        unet = self.build_unet_model()
        metrics = [keras.metrics.BinaryIoU(target_class_ids=[0, 1], threshold=0.5)]
        unet.compile(optimizer=Adam(), loss=keras.losses.BinaryCrossentropy(from_logits=False),
                             metrics=metrics)
        unet.load_weights(self.weights_path)
        self.unet = unet
        return

    def predict(self, image):
        #predictions = self.unet.predict(tf.stack(image))
        model_input = self.input_tensor_preprocessing(image)
        predictions = self.unet.predict(model_input)
        predictions = tf.squeeze(predictions)
        predictions = predictions.numpy() > 0.5
        return predictions
