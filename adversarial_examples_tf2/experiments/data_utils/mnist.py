import tensorflow as tf


def load_mnist(batch_size):
    """
    Helper function which loads the MNIST dataset for training and testing.

    Args:
        batch_size (int): the batch size for the training and testing datasets.

    Returns:
        train_data (tf.data.Dataset): MNIST training dataset with shape (None,
            28, 28) and the specified batch size.
        test_data (tf.data.Dataset): MNIST testing dataset with shape (None,
            28, 28) and the specified batch size.

    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_data = train_data.batch(batch_size)
    test_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_data = test_data.batch(batch_size)
    return train_data, test_data
