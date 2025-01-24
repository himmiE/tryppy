import json
import os
import keras
from keras import layers
import tensorflow as tf
from keras.src.optimizers import Adam


class Model:
    def __init__(self, model_name="default", verbose=1):
        current_dir = os.path.abspath(__file__).parent.resolve()
        config_path = current_dir + r"/resources/" + model_name + ".json"
        self.model_name = model_name
        self.config = None
        self.weights_path = ""
        self.model = None

        if verbose > 1:
            if os.path.isfile(config_path):
                print(f"Found corresponding config-file for model: {self.model_name}")
                with open(config_path) as f:
                    json_data = json.load(f)
                self.config = json_data
                self.weights_path = json_data["weights"]

            else:
                print(f"No corresponding config-file found for model: {self.model_name}.")
                print(f"Please ensure that the desired config-file is at the given location: {config_path}"
                      f"Please ensure that the model_name is correct."
                      f"If you are not using your own config file for training, leave model_name at default.")


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

    def load_model(self):
        model = self.build_unet_model()
        metrics = [keras.metrics.BinaryIoU(target_class_ids=[0, 1], threshold=0.5)]
        model.compile(optimizer=Adam(), loss=keras.losses.BinaryCrossentropy(from_logits=False),
                             metrics=metrics)
        model.load_weights(self.weights_path)
        self.model = model
        return

    def predict(self, images):
        predictions = self.model.predict(tf.stack(images))
        predictions = predictions > 0.5
        return predictions
