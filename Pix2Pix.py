class ResidualLayer(K.layers.Layer):
    def __init__(self, ffnn, **kwargs):
        super().__init__(**kwargs)
        self.ffnn = K.models.Sequential(ffnn)

    def call(self, inputs, *args, **kwargs):
        return tf.concat((self.ffnn(inputs), inputs), axis=-1)


def create_discriminator_p2p():
    inputs = K.layers.Input(shape=(128, 128, 3))
    conditioning = K.layers.Input(shape=(128, 128, 1))
    x = K.layers.Concatenate(axis=-1)([inputs, conditioning])
    x = K.layers.BatchNormalization(axis=-1)(x)
    x = K.layers.Conv2D(512, 3, activation="linear", use_bias=False)(x)
    x = K.layers.BatchNormalization(axis=-1)(x)
    x = K.layers.Activation(tf.nn.leaky_relu)(x)
    x = K.layers.Conv2D(128, 1, activation="linear", use_bias=False)(x)
    x = K.layers.BatchNormalization(axis=-1)(x)
    x = K.layers.Activation(tf.nn.leaky_relu)(x)
    x = K.layers.Conv2D(128, 3, activation="linear", use_bias=False)(x)
    x = K.layers.BatchNormalization(axis=-1)(x)
    x = K.layers.Activation(tf.nn.sigmoid)(x)
    x = K.layers.Conv2D(64, 1, activation="linear", use_bias=False)(x)
    x = K.layers.BatchNormalization(axis=-1)(x)
    x = K.layers.Activation(tf.nn.leaky_relu)(x)
    x = K.layers.Conv2D(64, 3, activation="linear", use_bias=False)(x)
    x = K.layers.BatchNormalization(axis=-1)(x)
    x = K.layers.Activation(tf.nn.leaky_relu)(x)
    x = K.layers.Conv2D(1, 1, activation="linear", use_bias=False)(x)
    x = K.layers.BatchNormalization(axis=-1)(x)
    x = K.layers.Activation(tf.nn.sigmoid)(x)
    disc = K.models.Model(inputs=[conditioning, inputs], outputs=x)
    return disc


def create_generator_p2p():
    inputs = K.layers.Input(shape=(128, 128, 1))
    x = K.layers.BatchNormalization(axis=-1)(inputs)
    x = K.layers.Conv2D(16, 3, activation="linear", padding="SAME", use_bias=False)(x)
    x = K.layers.BatchNormalization(axis=-1)(x)
    x = K.layers.Activation(tf.nn.leaky_relu)(x)
    x = K.layers.Conv2D(16, 3, activation="linear", padding="SAME", use_bias=False)(x)
    x = K.layers.BatchNormalization(axis=-1)(x)
    x = K.layers.Activation(tf.nn.leaky_relu)(x)
    x = K.layers.MaxPool2D(2)(x)
    x = ResidualLayer([
        K.layers.Conv2D(32, 3, activation="linear", padding="SAME", use_bias=False),
        K.layers.BatchNormalization(axis=-1),
        K.layers.Activation(tf.nn.leaky_relu),
        K.layers.Conv2D(32, 3, activation="linear", padding="SAME", use_bias=False),
        K.layers.BatchNormalization(axis=-1),
        K.layers.Activation(tf.nn.leaky_relu),
        K.layers.MaxPool2D(2),
        ResidualLayer([
            K.layers.Conv2D(64, 3, activation="linear", padding="SAME", use_bias=False),
            K.layers.BatchNormalization(axis=-1),
            K.layers.Activation(tf.nn.leaky_relu),
            K.layers.Conv2D(64, 3, activation="linear", padding="SAME", use_bias=False),
            K.layers.BatchNormalization(axis=-1),
            K.layers.Activation(tf.nn.leaky_relu),
            K.layers.MaxPool2D(2),
            ResidualLayer([
                K.layers.Conv2D(128, 3, activation="linear", padding="SAME", use_bias=False),
                K.layers.BatchNormalization(axis=-1),
                K.layers.Activation(tf.nn.leaky_relu),
                K.layers.Conv2D(128, 3, activation="linear", padding="SAME", use_bias=False),
                K.layers.BatchNormalization(axis=-1),
                K.layers.Activation(tf.nn.leaky_relu),
                K.layers.MaxPool2D(2),
                ResidualLayer([
                    K.layers.Conv2D(256, 1, activation="linear", padding="SAME", use_bias=False),
                    K.layers.BatchNormalization(axis=-1),
                    K.layers.Activation(tf.nn.leaky_relu),
                    K.layers.Conv2D(256, 3, activation="linear", padding="SAME", use_bias=False),
                    K.layers.BatchNormalization(axis=-1),
                    K.layers.Activation(tf.nn.leaky_relu),
                ]),
                K.layers.UpSampling2D(2),
                K.layers.Conv2D(128, 3, activation="linear", padding="SAME", use_bias=False),
                K.layers.BatchNormalization(axis=-1),
                K.layers.Activation(tf.nn.leaky_relu),
                K.layers.Conv2D(128, 3, activation="linear", padding="SAME", use_bias=False),
                K.layers.BatchNormalization(axis=-1),
                K.layers.Activation(tf.nn.leaky_relu),
            ]),
            K.layers.UpSampling2D(2),
            K.layers.Conv2D(64, 3, activation="linear", padding="SAME", use_bias=False),
            K.layers.BatchNormalization(axis=-1),
            K.layers.Activation(tf.nn.leaky_relu),
            K.layers.Conv2D(64, 3, activation="linear", padding="SAME", use_bias=False),
            K.layers.BatchNormalization(axis=-1),
            K.layers.Activation(tf.nn.leaky_relu),
        ]),
        K.layers.UpSampling2D(2),
        K.layers.Conv2D(32, 3, activation="linear", padding="SAME", use_bias=False),
        K.layers.BatchNormalization(axis=-1),
        K.layers.Activation(tf.nn.leaky_relu),
        K.layers.Conv2D(32, 3, activation="linear", padding="SAME", use_bias=False),
        K.layers.BatchNormalization(axis=-1),
        K.layers.Activation(tf.nn.leaky_relu),
    ])(x)
    x = K.layers.UpSampling2D(2)(x)
    x = K.layers.Conv2D(16, 3, activation="linear", padding="SAME", use_bias=False)(x)
    x = K.layers.BatchNormalization(axis=-1)(x)
    x = K.layers.Activation(tf.nn.leaky_relu)(x)
    x = K.layers.Conv2D(16, 3, activation="linear", padding="SAME", use_bias=False)(x)
    x = K.layers.BatchNormalization(axis=-1)(x)
    x = K.layers.Activation(tf.nn.leaky_relu)(x)
    x = K.layers.Conv2D(3, 3, activation="sigmoid", padding="SAME")(x)

    unet = K.models.Model(inputs=inputs, outputs=x)
    return unet


if __name__ == "__main__":
    with open('drive/MyDrive/cache/X_train.npy', 'rb') as f:
        X_train = np.load(f)
    with open('drive/MyDrive/cache/X_val.npy', 'rb') as f:
        X_val = np.load(f)
    with open('drive/MyDrive/cache/X_test.npy', 'rb') as f:
        X_test = np.load(f)

    unet = create_generator_p2p()
    patch_gan = create_discriminator_p2p()

    unet.compile(
        optimizer=tf.optimizers.legacy.Adam(1e-4, beta_1=0.5),
        loss=tf.losses.MeanAbsoluteError()
    )
    patch_gan.compile(
        optimizer=tf.optimizers.legacy.Adam(1e-4, beta_1=0.5)
    )

    train_pix2pix(X_train, unet, patch_gan, "pix2pix-0.01", alpha=0.01, save_only_weights=False,
                  batch_size=8)
