import numpy
from sklearn.model_selection import train_test_split


def convert_features_to_pairwise(training_feature_matrix, validation_feature_matrix, normalize_rows=True, standardize=True):
    
    num_training_examples = training_feature_matrix.shape[0]
    num_validation_examples = training_feature_matrix.shape[0]
    num_words = training_feature_matrix.shape[1]
    
    pairwise_training_feature_matrix = numpy.zeros((num_training_examples, num_words + num_words * num_words))
    pairwise_validation_feature_matrix = numpy.zeros((num_validation_examples, num_words + num_words * num_words))
    
    for example_index in range(len(training_feature_matrix)):
        for word_1_index in range(num_words):
            pairwise_training_feature_matrix[example_index][word_1_index * (num_words + 1)]
            for word_2_index in range(num_words):
                pairwise_training_feature_matrix[example_index][word_1_index * (num_words + 1) + word_2_index + 1] = \
                    training_feature_matrix[example_index][word_1_index] * training_feature_matrix[example][word_2_index]
    
    for example_index in range(len(validation_feature_matrix)):
        for word_1_index in range(num_words):
            pairwise_validation_feature_matrix[example_index][word_1_index * (num_words + 1)]
            for word_2_index in range(num_words):
                pairwise_validation_feature_matrix[example_index][word_1_index * (num_words + 1) + word_2_index + 1] = \
                    validation_feature_matrix[example_index][word_1_index] * validation_feature_matrix[example][word_2_index]
    
    non_zero_columns = ~numpy.all((pairwise_training_feature_matrix==0), axis=0)
    pairwise_training_feature_matrix = pairwise_training_feature_matrix[:, non_zero_columns]
    pairwise_validation_feature_matrix = pairwise_validation_feature_matrix[:, non_zero_columns]
    
    # Normalize the rows to get frequency of word pairs
    pairwise_training_feature_matrix = numpy.divide(pairwise_training_feature_matrix,
                                                    pairwise_training_feature_matrix.sum(axis=0))
    pairwise_validation_feature_matrix = numpy.divide(pairwise_validation_feature_matrix,
                                                      pairwise_validation_feature_matrix.sum(axis=0))
    
    if standardize:
        
        # Let's see some statistics
        mean = pairwise_training_feature_matrix.mean(axis=0).mean()
        SD = pairwise_training_feature_matrix.std(axis=0).std()
        # Find pixels with no information and remove them from the dataset
        zero_pixels = numpy.where(SD == 0)[0]
        SD = numpy.delete(SD, zero_pixels)
        pairwise_training_feature_matrix = numpy.delete(pairwise_training_feature_matrix, zero_pixels, axis=1)
        pairwise_validation_feature_matrix = numpy.delete(pairwise_validation_feature_matrix, zero_pixels, axis=1)
        mean = numpy.delete(mean, zero_pixels)

        # Standardize the data based on the statistics of the training set
        pairwise_training_feature_matrix = numpy.subtract(pairwise_training_feature_matrix, mean)
        pairwise_training_feature_matrix = numpy.divide(pairwise_training_feature_matrix, SD)

        pairwise_validation_feature_matrix = numpy.subtract(pairwise_validation_feature_matrix, mean)
        pairwise_validation_feature_matrix = numpy.divide(pairwise_validation_feature_matrix, SD)
    
    return pairwise_training_feature_matrix, pairwise_validation_feature_matrix

def get_headers():
    """
    Loads the header from the training data
    
    :return: headers
    """
    file = open("data/training_data.txt")
    headers = file.readline()
    headers = headers.split(' ')
    headers = headers[1:]
    file.close()
    return headers

def get_training_data(validation_size=0.3, onehot=1, standardize=False, normalize_rows=False):
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

    file = open("data/training_data.txt")

    words = file.readline().split(" ")
    words[-1] = words[-1].rstrip()
    file.close()

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
            
    if normalize_rows:
        training_features = numpy.divide(training_features, training_features.sum(axis=0))
        
        if validation_features is not None:
            validation_features = numpy.divide(validation_features, validation_features.sum(axis=0))

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


def get_test_data(standardize=False, normalize_rows=False):
    """
    Loads the test data
    :param standardize: Whether to standardize the data by mu and sd of the
        training set
    :return testing_features: the feature matrix
    """

    data = numpy.loadtxt(
        "data/test_data.txt", skiprows=1, delimiter=" ")

    testing_features = data
    
    if normalize_rows:
        testing_features = numpy.divide(testing_features, testing_features.sum(axis=0))

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
