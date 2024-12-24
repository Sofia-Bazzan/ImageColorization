class ResidualLayer(K.layers.Layer):
    def __init__(self, ffnn, **kwargs):
        super().__init__(**kwargs)
        self.ffnn = K.models.Sequential(ffnn)

    def call(self, inputs, *args, **kwargs):
        return self.ffnn(inputs) + inputs


def create_resnet():
    inputs = K.layers.Input(shape=(128, 128, 1))
    x = K.layers.BatchNormalization(axis=-1)(inputs)
    x = K.layers.Conv2D(256, 1, activation="linear", padding="SAME")(x)

    # Initial Residual Block
    x = ResidualLayer([
        K.layers.BatchNormalization(axis=-1),
        K.layers.Activation(tf.nn.leaky_relu),
        K.layers.Conv2D(32, 1, activation="linear", padding="SAME", use_bias=False),
        K.layers.BatchNormalization(axis=-1),
        K.layers.Activation(tf.nn.leaky_relu),
        K.layers.Conv2D(32, 3, activation="linear", padding="SAME", use_bias=False),
        K.layers.BatchNormalization(axis=-1),
        K.layers.Activation(tf.nn.leaky_relu),
        K.layers.Conv2D(256, 1, activation="linear", padding="SAME", use_bias=False),
    ])(x)

    # Residual Blocks
    for _ in range(20):
        x = ResidualLayer([
            K.layers.BatchNormalization(axis=-1),
            K.layers.Activation(tf.nn.leaky_relu),
            K.layers.Conv2D(32, 1, activation="linear", padding="SAME", use_bias=False),
            K.layers.BatchNormalization(axis=-1),
            K.layers.Activation(tf.nn.leaky_relu),
            K.layers.Conv2D(32, 3, activation="linear", padding="SAME", use_bias=False),
            K.layers.BatchNormalization(axis=-1),
            K.layers.Activation(tf.nn.leaky_relu),
            K.layers.Conv2D(256, 1, activation="linear", padding="SAME", use_bias=False),
        ])(x)

    # Finalization
    x = K.layers.BatchNormalization(axis=-1)(x)
    x = K.layers.Activation(tf.nn.leaky_relu)(x)
    x = K.layers.Conv2D(3, 3, activation="sigmoid", padding="SAME")(x)

    resnet = K.models.Model(inputs=inputs, outputs=x)
    return resnet


if __name__ == "__main__":
    with open('drive/MyDrive/dataset/X_train.npy', 'rb') as f:
        X_train = np.load(f)
    with open('drive/MyDrive/dataset/X_val.npy', 'rb') as f:
        X_val = np.load(f)
    with open('drive/MyDrive/dataset/X_test.npy', 'rb') as f:
        X_test = np.load(f)

    resnet = create_resnet()

    resnet.compile(
        optimizer=tf.optimizers.legacy.Adam(1e-3),
        loss=tf.losses.MeanSquaredError()
    )
    train_dnn(X_train, X_val, resnet, "resnet", batch_size=8)
