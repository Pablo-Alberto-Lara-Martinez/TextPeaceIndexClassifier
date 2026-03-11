# Model Architecture and Theoretical Background

This document outlines the theoretical foundations of the Neural Network architecture used in this repository to classify the peace index of countries based on their textual embeddings.

## 1. The 1D Convolutional Neural Network (1D CNN)

While Convolutional Neural Networks (CNNs) are traditionally associated with image processing (2D), 1D CNNs are highly effective for sequence data, including natural language. In this context, the text has already been transformed into dense vector representations (embeddings) using ChromaDB. 

The 1D CNN processes these embeddings to automatically extract hierarchical spatial patterns—essentially identifying localized semantic features (similar to n-grams) that are highly indicative of a country's peace classification.

### Layer-by-Layer Breakdown

Our architecture is built using TensorFlow/Keras with the following sequence of layers:

* **`Conv1D` (Convolutional Layer):** * *Function:* Applies sliding filters (kernels) across the input sequence of embeddings.
    * *Details:* We use 64 filters with a kernel size of 3 and a ReLU (Rectified Linear Unit) activation function. This layer acts as a feature extractor, learning to recognize specific local patterns in the vector space.
* **`MaxPooling1D` (Pooling Layer):** * *Function:* Reduces the dimensionality of the feature maps extracted by the convolutional layer.
    * *Details:* With a pool size of 2, it downsamples the input by taking the maximum value over small windows. This retains the most prominent features while reducing computational load and helping to prevent overfitting by providing translation invariance.
* **`Flatten` Layer:** * *Function:* Transforms the 2D matrix output from the pooling layer into a single continuous 1D vector, preparing it for the fully connected neural network layers.
* **`Dense` Layers (Fully Connected):** * *Function:* Learns complex, non-linear combinations of the features extracted by the convolutional layers.
    * *Details:* The network uses two hidden dense layers (128 and 64 neurons, respectively) with ReLU activation. 
* **Final Output Layer:** * *Function:* Outputs the final prediction.
    * *Details:* A dense layer with a single neuron using a `sigmoid` activation function. The sigmoid function maps the raw output to a value between 0 and 1, representing the probability that the given text belongs to the positive peace class (1).

## 2. Loss Function: Binary Crossentropy

To train the network, we evaluate its performance using **Binary Crossentropy** (also known as log loss). Since our goal is a binary classification (Peaceful vs. Non-Peaceful), this is the standard and most mathematically sound loss function.

It measures the performance of a classification model where the prediction input is a probability value between 0 and 1. The loss increases as the predicted probability diverges from the actual label. 

Mathematically, for a set of $N$ samples, the binary crossentropy loss $L$ is calculated as:

$$L = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

Where:
* $y_i$ is the true label (0 or 1) for the $i$-th sample.
* $\hat{y}_i$ is the predicted probability that the $i$-th sample belongs to class 1.

The `adam` optimizer is utilized to iteratively adjust the network's weights to minimize this loss function during training.