import tensorflow as tf


def input_pipeline(batch_size=32):

    # Load mnist
    mnist = tf.keras.datasets.mnist
    # (10000, 28, 28) * 0.7
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Add a channels dimension
    x_train = x_train[..., tf.newaxis].astype("float32")
    x_test = x_test[..., tf.newaxis].astype("float32")

    # Create train & test datasets
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    # Shuffle and batch
    train_ds = train_ds.shuffle(10000).batch(batch_size)
    test_ds = test_ds.batch(batch_size)

    return train_ds, test_ds


if __name__ == "__main__":
    train_ds, test_ds = input_pipeline()
