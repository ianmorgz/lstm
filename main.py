import numpy as np
from dataset import generate_sine_data
from lstm import LSTM

INPUT_FEATURES = 1

if __name__ == "__main__":
    dataset = generate_sine_data(length=100, step = 0.5)
    X_train, train_series, X_test, test_series = dataset


