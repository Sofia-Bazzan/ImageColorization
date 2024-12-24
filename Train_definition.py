#Import packages
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import trange
import os
from PIL import Image
import tensorflow as tf
from tqdm import tqdm
import keras.api._v2.keras as K
import pickle
import time

#connection to drive
from google.colab import drive
drive.mount('/content/drive')

#definition of train and loss functions

def train_dnn(X_train, X_val, model, name, batch_size=128, epochs=30, save_only_weights=False):
    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            preds = model(x, training=True)
            loss_value = model.loss(y, preds)
        grads = tape.gradient(loss_value, model.trainable_weights)
        model.optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return loss_value

    def hue_shift(image):
        hue_shift_value = tf.random.uniform([], -0.5, 0.5)
        adjusted_image = tf.image.adjust_hue(image, hue_shift_value)
        return adjusted_image

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, X_train))
    train_dataset = train_dataset.map(lambda x, y: (hue_shift(x), y))

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, X_train)) \
        .map(lambda x, y: (tf.reduce_mean(x, axis=-1, keepdims=True), tf.cast(y, tf.float32) / 255.)) \
        .cache() \
        .shuffle(buffer_size=1024) \
        .batch(batch_size)

    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, X_val)) \
        .map(lambda x, y: (tf.reduce_mean(x, axis=-1, keepdims=True), tf.cast(y, tf.float32) / 255.)) \
        .cache() \
        .shuffle(buffer_size=1024) \
        .batch(batch_size)

    losses_history = []
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        start_time = time.time()

        train_losses = []
        for step, (x_batch_train, y_batch_train) in enumerate(tqdm(train_dataset)):
            train_losses.append(
                train_step(x_batch_train, y_batch_train)
            )

        val_losses = []
        for x_batch_val, y_batch_val in val_dataset:
            val_losses.append(model.evaluate(x_batch_val, y_batch_val, verbose=0))

        print("Train loss: %.4f" % (float(np.mean(train_losses)),))
        print("Validation loss: %.4f" % (float(np.mean(val_losses)),))
        print("Time taken: %.2fs" % (time.time() - start_time))

        losses_history.append([float(np.mean(train_losses)), float(np.mean(val_losses))])

        if save_only_weights:
            weights_dir = f"drive/MyDrive/models/{name}"
            os.makedirs(weights_dir, exist_ok=True)
            model.save_weights(f"{weights_dir}/{epoch}")
        else:
            model_dir = f"drive/MyDrive/models/{name}"
            os.makedirs(model_dir, exist_ok=True)
            model.save(f"{model_dir}/{epoch}")

        history_dir = f"drive/MyDrive/models/{name}"
        os.makedirs(history_dir, exist_ok=True)
        with open(f"{history_dir}/loss_history", "wb+") as f:
            pickle.dump(losses_history, f)

    return losses_history


def train_pix2pix(X_train, generator, discriminator, name, alpha=1.0, batch_size=32, epochs=30, save_only_weights=False):
    @tf.function
    def train_step(x, y):
        generated_rgb_images = generator(x, training=False)
        with tf.GradientTape() as tape:
            preds_1 = discriminator([x, y])
            preds_0 = discriminator([x, generated_rgb_images])
            loss = tf.reduce_mean((1 - preds_1)**2) + tf.reduce_mean(preds_0**2)
        grads = tape.gradient(loss, discriminator.trainable_weights)
        discriminator.optimizer.apply_gradients(zip(grads, discriminator.trainable_weights))

        with tf.GradientTape() as tape:
            generated_rgb_images = generator(x)
            preds = discriminator([x, generated_rgb_images])
            loss = generator.loss(y, generated_rgb_images) + alpha * (
                -tf.reduce_mean(
                    tf.math.log(preds + 1e-8)
                )
            )
        grads = tape.gradient(loss, generator.trainable_weights)
        generator.optimizer.apply_gradients(zip(grads, generator.trainable_weights))
        return tf.reduce_mean(preds)

    def hue_shift(image):
        hue_shift_value = tf.random.uniform([], -0.5, 0.5)
        adjusted_image = tf.image.adjust_hue(image, hue_shift_value)
        return adjusted_image

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, X_train))
    train_dataset = train_dataset.map(lambda x, y: (hue_shift(x), y))


     #def contrast_adjust(image):
      #contrast_factor = tf.random.uniform([], 0.5, 1.5)
      #adjusted_image = tf.image.adjust_contrast(image, contrast_factor)
      #return adjusted_image

    #train_dataset = tf.data.Dataset.from_tensor_slices((X_train, X_train))
    #train_dataset = train_dataset.map(lambda x, y: (contrast_adjust(x), y))

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, X_train)) \
        .map(lambda x, y: (tf.reduce_mean(x, axis=-1, keepdims=True), tf.cast(y, tf.float32) / 255.)) \
        .cache() \
        .shuffle(buffer_size=1024) \
        .batch(batch_size)

    for epoch in range(epochs):
        disc_losses = []
        print("\nStart of epoch %d" % (epoch,))
        start_time = time.time()

        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            disc_losses.append(train_step(x_batch_train, y_batch_train))
            print(f"\r{step}/{train_dataset.cardinality().numpy()} - Avg disc pred: {np.mean(disc_losses[-3:])}", end="")

        print("Time taken: %.2fs" % (time.time() - start_time))

        if save_only_weights:
            generator.save_weights(f"models/{name}/{epoch}/generator")
            discriminator.save_weights(f"models/{name}/{epoch}/discriminator")
        else:
            generator.save(f"models/{name}/{epoch}/generator")
            discriminator.save(f"models/{name}/{epoch}/discriminator")
