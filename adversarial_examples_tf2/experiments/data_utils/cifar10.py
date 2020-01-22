import tensorflow as tf


def load_cifar10(batch_size):
    """
    Helper function which loads the CIFAR10 dataset for training and testing.

    Args:
        batch_size (int): the batch size for the training and testing datasets.

    Returns:
        train_data (tf.data.Dataset): CIFAR10 training dataset with shape (None,
            32, 32) and the specified batch size.
        test_data (tf.data.Dataset): CIFAR10 testing dataset with shape (None,
            32, 32) and the specified batch size.

    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    # Normalize pixel values to be between 0 and 1.
    x_train, x_test = x_train / 255.0, x_test / 255.0

    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_data = train_data.batch(batch_size)
    test_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_data = test_data.batch(batch_size)
    return train_data, test_data
