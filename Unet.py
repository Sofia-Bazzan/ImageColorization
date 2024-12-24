class ResidualLayer(K.layers.Layer):
    def __init__(self, ffnn, **kwargs):
        super().__init__(**kwargs)
        self.ffnn = K.models.Sequential(ffnn)

    def call(self, inputs, *args, **kwargs):
        return tf.concat((self.ffnn(inputs), inputs), axis=-1)


def create_unet():
    inputs = K.layers.Input(shape=(128, 128, 1))
    x = K.layers.BatchNormalization(axis=-1)(inputs)

    # Encoder
    for filters in [16, 32, 64, 128, 256]:
        x = K.layers.Conv2D(filters, 3, activation="linear", padding="SAME", use_bias=False)(x)
        x = K.layers.BatchNormalization(axis=-1)(x)
        x = K.layers.Activation(tf.nn.leaky_relu)(x)

        if filters != 256:
            x = K.layers.MaxPool2D(2)(x)
        else:
            x = ResidualLayer([
                K.layers.Conv2D(256, 1, activation="linear", padding="SAME", use_bias=False),
                K.layers.BatchNormalization(axis=-1),
                K.layers.Activation(tf.nn.leaky_relu),
                K.layers.Conv2D(256, 3, activation="linear", padding="SAME", use_bias=False),
                K.layers.BatchNormalization(axis=-1),
                K.layers.Activation(tf.nn.leaky_relu),
            ])(x)

    # Decoder
    for filters in [128, 64, 32, 16]:
        x = K.layers.UpSampling2D(2)(x)
        x = K.layers.Conv2D(filters, 3, activation="linear", padding="SAME", use_bias=False)(x)
        x = K.layers.BatchNormalization(axis=-1)(x)
        x = K.layers.Activation(tf.nn.leaky_relu)(x)
        x = K.layers.Conv2D(filters, 3, activation="linear", padding="SAME", use_bias=False)(x)
        x = K.layers.BatchNormalization(axis=-1)(x)
        x = K.layers.Activation(tf.nn.leaky_relu)(x)

    # Output Layer
    x = K.layers.UpSampling2D(2)(x)
    x = K.layers.Conv2D(3, 3, activation="sigmoid", padding="SAME")(x)

    unet = K.models.Model(inputs=inputs, outputs=x)
    return unet


if __name__ == "__main__":
    with open('drive/MyDrive/dataset/X_train.npy', 'rb') as f:
        X_train = np.load(f)
    with open('drive/MyDrive/dataset/X_val.npy', 'rb') as f:
        X_val = np.load(f)
    with open('drive/MyDrive/dataset/X_test.npy', 'rb') as f:
        X_test = np.load(f)
    unet = create_unet()
    unet.compile(
        optimizer=tf.optimizers.legacy.Adam(1e-3),
        loss=tf.losses.MeanSquaredError()
    )

    train_dnn(X_train, X_val, unet, "unet", batch_size=16, save_only_weights=False)
