from cgi import test
from tabnanny import verbose
import tensorflow as tf
from keras.layers import Dense, Flatten, Conv2D, Dropout, Softmax
from keras import Model
from keras.utils import Progbar
from datetime import datetime


class simpleNet(Model):
    def __init__(self):
        super(simpleNet, self).__init__()
        self.conv1 = Conv2D(64, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128)
        self.d2 = Dense(10)
        self.sftm = Softmax()

        self.loss_object = None
        self.optimizer = None
        self.train_loss = None
        self.train_accuracy = None
        self.test_loss = None
        self.test_accuracy = None

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        x = self.sftm(x)
        return x

    def compile(self):
        # Loss & optimizer
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam()

        # Metrics
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='train_accuracy')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='test_accuracy')

    @tf.function
    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            predictions = self(images, training=True)
            loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(labels, predictions)

    @tf.function
    def test_step(self, images, labels):
        predictions = self(images, training=False)
        t_loss = self.loss_object(labels, predictions)

        self.test_loss(t_loss)
        self.test_accuracy(labels, predictions)

    def fit(self, epochs, train_ds, test_ds, verbose):
        train_summary_writer, test_summary_writer = self.tb_callback()
        for epoch in range(epochs):
            step = 0
            progbar = Progbar(target=int(
                train_ds.cardinality()), verbose=verbose)

            for images, labels in train_ds:
                step += 1
                self.train_step(images, labels)
                progbar.update(step)
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', self.train_loss.result(), step=epoch)
                tf.summary.scalar(
                    'accuracy', self.train_accuracy.result(), step=epoch)

            for test_images, test_labels in test_ds:
                self.test_step(test_images, test_labels)
            with test_summary_writer.as_default():
                tf.summary.scalar('loss', self.test_loss.result(), step=epoch)
                tf.summary.scalar(
                    'accuracy', self.test_accuracy.result(), step=epoch)

            if verbose == 1:
                template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
                print(template.format(epoch+1,
                                      self.train_loss.result(),
                                      self.train_accuracy.result()*100,
                                      self.test_loss.result(),
                                      self.test_accuracy.result()*100))

            # Reset the metrics at the start of the next epoch
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.test_loss.reset_states()
            self.test_accuracy.reset_states()

    def tb_callback(self):
        current_time = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        train_log_dir = 'logs/' + str(type(self).__name__) + '/train'
        test_log_dir = 'logs/' + str(type(self).__name__) + '/test'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        test_summary_writer = tf.summary.create_file_writer(test_log_dir)
        return train_summary_writer, test_summary_writer


class sequentialNet():
    def __init__(self):
        self.conv1 = Conv2D(64, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128)
        self.d2 = Dense(10)
        self.sft = Softmax()
        self.model = tf.keras.Sequential(
            [self.conv1,
             self.flatten,
             self.d1,
             self.d2,
             self.sft])

    def compile(self):
        self.model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=["accuracy"])

    def fit(self, epochs, train_ds, test_ds, verbose):
        self.model.fit(train_ds, epochs=epochs,
                       verbose=verbose, validation_data=test_ds)


class simpleNet_graph(Model):
    def __init__(self):
        super(simpleNet_graph, self).__init__()
        self.conv1 = Conv2D(64, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128)
        self.d2 = Dense(10)
        self.sftm = Softmax()

        self.loss_object = None
        self.optimizer = None
        self.train_loss = None
        self.train_accuracy = None
        self.test_loss = None
        self.test_accuracy = None

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        x = self.sftm(x)
        return x

    def compile(self):
        # Loss & optimizer
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam()

        # Metrics
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='train_accuracy')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='test_accuracy')

    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            predictions = self(images, training=True)
            loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(labels, predictions)

    def test_step(self, images, labels):
        predictions = self(images, training=False)
        t_loss = self.loss_object(labels, predictions)

        self.test_loss(t_loss)
        self.test_accuracy(labels, predictions)

    def fit(self, epochs, train_ds, test_ds, verbose):
        train_summary_writer, test_summary_writer = self.tb_callback()
        for epoch in range(epochs):
            step = 0
            progbar = Progbar(target=int(
                train_ds.cardinality()), verbose=verbose)

            for images, labels in train_ds:
                step += 1
                self.train_step(images, labels)
                progbar.update(step)
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', self.train_loss.result(), step=epoch)
                tf.summary.scalar(
                    'accuracy', self.train_accuracy.result(), step=epoch)

            for test_images, test_labels in test_ds:
                self.test_step(test_images, test_labels)
            with test_summary_writer.as_default():
                tf.summary.scalar('loss', self.test_loss.result(), step=epoch)
                tf.summary.scalar(
                    'accuracy', self.test_accuracy.result(), step=epoch)

            if verbose == 1:
                template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
                print(template.format(epoch+1,
                                      self.train_loss.result(),
                                      self.train_accuracy.result()*100,
                                      self.test_loss.result(),
                                      self.test_accuracy.result()*100))

            # Reset the metrics at the start of the next epoch
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.test_loss.reset_states()
            self.test_accuracy.reset_states()

    def tb_callback(self):
        current_time = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        train_log_dir = 'logs/' + str(type(self).__name__) + '/train'
        test_log_dir = 'logs/' + str(type(self).__name__) + '/test'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        test_summary_writer = tf.summary.create_file_writer(test_log_dir)
        return train_summary_writer, test_summary_writer
