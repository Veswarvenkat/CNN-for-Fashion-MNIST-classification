# CNN for Fashion MNIST Classification

This project uses TensorFlow and Keras to build and train a Convolutional Neural Network (CNN) for classifying grayscale images from the Fashion MNIST dataset. The model learns to identify 10 different types of fashion apparel.

## Project Overview

The notebook covers the complete workflow for an image classification task:
* **Data Loading:** Loads the built-in Fashion MNIST dataset from `tf.keras.datasets`.
* **Preprocessing:** Reshapes the images to include a channel dimension (`28x28x1`) and normalizes the pixel values to be between 0 and 1.
* **Model Building:** Defines a `Sequential` CNN model.
* **Training:** Trains the model on the training data for 10 epochs.
* **Evaluation:** Evaluates the trained model's performance on the unseen test dataset.
* **Visualization:** Plots the training and validation accuracy over time.

## Dataset: Fashion MNIST

The dataset consists of 70,000 grayscale images (60,000 for training, 10,000 for testing), each 28x28 pixels. There are 10 distinct classes:

| Label | Class |
| :--- | :--- |
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

---

## Model Architecture

The model is a `Sequential` stack of layers:

1.  **Conv2D Block 1:**
    * `Conv2D` (32 filters, 3x3 kernel, `relu` activation)
    * `L2 Regularization` (0.001)
    * `MaxPooling2D` (2x2 pool size)

2.  **Conv2D Block 2:**
    * `Conv2D` (64 filters, 3x3 kernel, `relu` activation)
    * `L2 Regularization` (0.001)
    * `MaxPooling2D` (2x2 pool size)

3.  **Conv2D Block 3:**
    * `Conv2D` (128 filters, 3x3 kernel, `relu` activation)
    * `L2 Regularization` (0.001)
    * `MaxPooling2D` (2x2 pool size)

4.  **Classifier Head:**
    * `Flatten`: Converts the 3D feature maps into a 1D vector.
    * `Dense` (128 units, `relu` activation)
    * `Dropout` (0.5): A 50% dropout rate for regularization to prevent overfitting.
    * `Dense` (10 units, `softmax` activation): The output layer that provides a probability distribution across the 10 classes.

The model is compiled using:
* **Optimizer:** `adam`
* **Loss Function:** `sparse_categorical_crossentropy` (used because the labels are integers, not one-hot encoded).
* **Metrics:** `accuracy`

---

## Requirements

To run this project, you'll need the following Python libraries:

* `tensorflow`
* `numpy`
* `matplotlib`

You can install them using pip:
```bash
pip install tensorflow numpy matplotlib
