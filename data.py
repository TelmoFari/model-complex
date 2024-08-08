import numpy as np
from sklearn.preprocessing import StandardScaler

def generate_data(num_samples=1000):
    x_data = np.random.rand(num_samples, 10)
    y_data = np.random.randint(2, size=num_samples)
    return x_data, y_data

def preprocess_data(x_train, x_test):
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    return x_train, x_test
