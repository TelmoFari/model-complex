import time
from model import create_model, save_model
from data import generate_data, preprocess_data
from tuner import tune_model
from tensorflow.keras.callbacks import EarlyStopping

def main():
    iteration = 1
    while True:
        print(f"Starting iteration {iteration}...")
        x_train, y_train = generate_data()
        x_test, y_test = generate_data(num_samples=200)

        x_train, x_test = preprocess_data(x_train, x_test)

        best_hps = tune_model(x_train, y_train)

        model = create_model(best_hps)
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        history = model.fit(x_train, y_train, epochs=30, batch_size=32, verbose=2, validation_split=0.2, callbacks=[early_stopping])

        save_model(model)

        iteration += 1
        time.sleep(60)

if __name__ == "__main__":
    main()
