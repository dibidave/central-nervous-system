import numpy
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical


def get_training_data(validation_size=0.3, use_one_hot_encoding=True):
    """
    Loads the training data, splits it into training and validation sets, and
    converts the labels into 2-class one-hot encoding

    :param validation_size: What ratio of the dataset to use as training data
    :param use_one_hot_encoding: Whether to use one hot encoding
    :return: (training_features, training_labels,
        validation_features, validation_labels)
    """

    data = numpy.loadtxt(
        "data/training_data.txt", skiprows=1, delimiter=" ")

    features = data[:, 1:]
    labels = data[:, 0]

    training_features, validation_features, training_labels, validation_labels = \
        train_test_split(features, labels, test_size=validation_size)

    if use_one_hot_encoding:
        training_labels = to_categorical(training_labels)
        validation_labels = to_categorical(validation_labels)

    return (training_features, training_labels,
            validation_features, validation_labels)
