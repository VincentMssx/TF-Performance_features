import tensorflow as tf
from models import simpleNet
from input_pipeline import input_pipeline
import os
from time import time
from tqdm import tqdm
from keras.utils import Progbar
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Global variables
EPOCHS = 5

if __name__ == "__main__":

    # Load & create mnist dataset
    train_ds, test_ds = input_pipeline()

    # Create an instance of the model
    simpleNet = simpleNet()

    # Loss & optimizer
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True)
    optimizer = tf.keras.optimizers.Adam()

    # Metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='test_accuracy')

    # training & test
    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = simpleNet(images, training=True)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, simpleNet.trainable_variables)
        optimizer.apply_gradients(
            zip(gradients, simpleNet.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)

    @tf.function
    def test_step(images, labels):
        predictions = simpleNet(images, training=False)
        t_loss = loss_object(labels, predictions)

        test_loss(t_loss)
        test_accuracy(labels, predictions)

    def fit(epochs):
        for epoch in range(epochs):

            for images, labels in train_ds:
                train_step(images, labels)

            for test_images, test_labels in test_ds:
                test_step(test_images, test_labels)

            print(
                f'Epoch {epoch + 1}, '
                f'Loss: {train_loss.result()}, '
                f'Accuracy: {train_accuracy.result() * 100}, '
                f'Test Loss: {test_loss.result()}, '
                f'Test Accuracy: {test_accuracy.result() * 100}'
            )

            # Reset the metrics at the start of the next epoch
            train_loss.reset_states()
            train_accuracy.reset_states()
            test_loss.reset_states()
            test_accuracy.reset_states()

    # with tf.device("CPU:0"):
    #     start = time()
    #     fit(EPOCHS)
    #     end = time()
    #     print('CPU :', end-start)

    with tf.device("GPU:0"):
        start = time()
        fit(EPOCHS)
        end = time()
        print('GPU :', end-start)
        simpleNet.summary()
