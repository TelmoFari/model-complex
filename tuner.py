from keras_tuner import RandomSearch
from model import create_model

def tune_model(x_train, y_train):
    tuner = RandomSearch(
        create_model,
        objective='accuracy',
        max_trials=125,
        executions_per_trial=30,
        directory='model-complex-ia',
        project_name='model-complex-ia'
    )

    tuner.search(x_train, y_train, epochs=70, validation_split=0.2)
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"Best model configuration: {best_hps.values}")
    return best_hps
