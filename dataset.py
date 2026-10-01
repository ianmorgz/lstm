import numpy as np

def generate_sine_data(length, step=0.5):
    X_train = np.arange(0, length, step)
    y_train = np.sin(X_train)

    X_test = np.arange(length, 2*length, step)
    y_test = np.sin(X_test)

    n_features = 1

    train_series = y_train.reshape((len(y_train), n_features))
    test_series = y_test.reshape((len(y_test), n_features))


    return X_train, train_series, X_test, test_series