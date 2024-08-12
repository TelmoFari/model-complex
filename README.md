Complex Machine Learning Model Overview:

This project features a complex machine learning model designed for binary classification tasks. The model leverages a combination of convolutional, recurrent, and dense layers to achieve high accuracy and generalization.

Model Architecture Convolutional Layers:

Purpose: Extract spatial features from sequential data.

Configuration: 
* Number of layers: 1 to 6
* Filters: 32 to 128
* Kernel sizes: 3 or 5

Purpose: Capture temporal and sequential dependencies.

Configuration: 
* Number of layers: 2 to 7
* Units per layer: 50 to 150
* Type: LSTM or GRU
* Dropout rate: 0.3 to 0.5

Purpose: Process features and perform final classification.

Configuration:
* Number of layers: 7 to 10
* Units per layer: 64 to 256
* L2 Regularization: 0.01 to 0.1
* Dropout rate: 0.3 to 0.5

Activation: sigmoid for binary classification.

Requirements:
* numpy
* tensorflow
* scikit-learn
* keras-tuner

How to install:

1. Import the Project:
* Go to the [Reblit](https://replit.com)  page, click on the "Import from GitHub" option and enter the URL:  `https://github.com/TelmoFari/model-complex`

2. Install the dependencies
* Run the following command to install the dependencies:

`pip install -r requirements.txt`

Replit will probably prompt you for a run command. Use the command below:

`python model-complex-ia.py`

Then a run button will appear and click and the model will be trained. 

Model Evaluation and Saving:

During training, the model is evaluated and automatically saved in the model-complex-ia folder.

Configuration Hyperparameters: 

* Adjust the hyperparameters: directly in the create_model function.
* Training Intervals: The model training loop runs every minute. Modify the time.sleep(60) interval in model-complex-ia.py if needed.

License:

This project is licensed under the MIT License.

Acknowledgments:

* TensorFlow and Keras: For providing the deep learning framework that made the development of this model possible.
* Scikit-learn: For its indispensable machine learning tools.

Feel free to adapt this project to your needs. Making necessary changes to files





