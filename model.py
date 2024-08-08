import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.metrics import Precision, Recall, AUC

def create_model(hp):
    model = keras.Sequential()

    # Adicionando camadas convolucionais
    for i in range(hp.Int('num_conv_layers', 1, 6)):
        filters = hp.Int(f'filters_{i}', min_value=32, max_value=128, step=32)
        kernel_size = hp.Choice(f'kernel_size_{i}', values=[3, 5])
        model.add(layers.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', padding='same', input_shape=(10, 1)))
        model.add(layers.MaxPooling1D(pool_size=2))
        model.add(layers.BatchNormalization())

    # Adicionando camadas LSTM/GRU
    for i in range(hp.Int('num_rnn_layers', 2, 7)):
        units = hp.Int(f'rnn_units_{i}', min_value=50, max_value=150, step=50)
        if hp.Choice(f'rnn_type_{i}', values=['LSTM', 'GRU']) == 'LSTM':
            model.add(layers.LSTM(units=units, return_sequences=True, dropout=hp.Float(f'dropout_rnn_{i}', min_value=0.3, max_value=0.5, step=0.1)))
        else:
            model.add(layers.GRU(units=units, return_sequences=True, dropout=hp.Float(f'dropout_rnn_{i}', min_value=0.3, max_value=0.5, step=0.1)))

    # Camadas densas
    num_layers = hp.Int('num_dense_layers', 7, 10)
    for i in range(num_layers):
        units = hp.Int(f'units_dense_{i}', min_value=64, max_value=256, step=32)
        model.add(layers.Dense(units=units, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(hp.Float(f'l2_reg_{i}', min_value=0.01, max_value=0.1, step=0.01))))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(rate=hp.Float(f'dropout_dense_{i}', min_value=0.3, max_value=0.5, step=0.1)))

    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='LOG')),
        loss='binary_crossentropy',
        metrics=['accuracy', Precision(), Recall(), AUC()]
    )
    return model

def save_model(model, filename="model-complex-ia.h5"):
    model.save(filename)
    print(f"Model saved to {filename}")
