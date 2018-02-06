import numpy
from sklearn.model_selection import train_test_split


def get_training_data(validation_size=0.3, onehot=1, standardize=False):
    """
    Loads the training data, splits it into training and validation sets, and
    converts the labels into 2-class one-hot encoding

    :param validation_size: What percent of the dataset to hold out for
        validation
    :param onehot: Whether to use one hot encoding
    :param standardize: Whether to standardize the data by mu and sd
    :return: (training_features, training_labels,
        validation_features, validation_labels)
    """

    data = numpy.loadtxt(
        "data/training_data.txt", skiprows=1, delimiter=" ")

    features = data[:, 1:]
    labels = data[:, 0]

    if validation_size == 0.0:
        training_features = features
        training_labels = labels
        validation_features = None
        validation_labels = None
    else:
        training_features, validation_features, training_labels,\
            validation_labels = \
            train_test_split(features, labels, test_size=validation_size)

    if onehot:
        from keras.utils.np_utils import to_categorical
        training_labels = to_categorical(training_labels)
        if validation_labels is not None:
            validation_labels = to_categorical(validation_labels)

    if standardize:

        # Let's see some statistics
        mean = training_features.mean(axis=0).mean()
        SD = training_features.std(axis=0).std()
        # Find pixels with no information and remove them from the dataset
        zero_pixels = numpy.where(SD == 0)[0]
        SD = numpy.delete(SD, zero_pixels)
        training_features = numpy.delete(training_features, zero_pixels, axis=1)

        if validation_features is not None:
            validation_features = numpy.delete(validation_features, zero_pixels,
                                               axis=1)
        mean = numpy.delete(mean, zero_pixels)

        # Standardize the data based on the statistics of the training set
        training_features = numpy.subtract(training_features, mean)
        training_features = numpy.divide(training_features, SD)

        if validation_features is not None:
            validation_features = numpy.subtract(validation_features, mean)
            validation_features = numpy.divide(validation_features, SD)

    return (training_features, training_labels,
            validation_features, validation_labels)


def get_testing_data(standardize=False):
    """
    Loads the test data

    :param onehot: Whether to use one hot encoding
    :param standardize: Whether to standardize the data by mu and sd of the
        training set
    :return testing_features: the feature matrix
    """

    data = numpy.loadtxt(
        "data/test_data.txt", skiprows=1, delimiter=" ")

    testing_features = data

    if standardize:
        training_features, _, _, _ = \
            get_training_data(validation_size=0.0, onehot=1, standardize=False)
        # Let's see some statistics
        mean = training_features.mean(axis=0).mean()
        SD = training_features.std(axis=0).std()
        # Find pixels with no information and remove them from the dataset
        zero_pixels = numpy.where(SD == 0)[0]
        SD = numpy.delete(SD, zero_pixels)
        testing_features = numpy.delete(testing_features, zero_pixels, axis=1)
        mean = numpy.delete(mean, zero_pixels)

        # Standardize the data based on the statistics of the training set
        testing_features = numpy.subtract(testing_features, mean)
        testing_features = numpy.divide(testing_features, SD)

    return testing_features


def save_prediction(labels, filename="data/test_labels.txt"):

    file = open(filename, "w")

    file.write("Id,Prediction\n")

    for label_index, label in enumerate(labels):
        file.write("%i,%i\n" % (label_index + 1, label))

    file.close()
