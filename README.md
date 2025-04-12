# Traffic Sign Detection using GTSRB

Welcome to **Traffic Sign Detection**! This project involves building a Convolutional Neural Network (CNN) model to detect and classify traffic signs using the German Traffic Sign Recognition Benchmark (GTSRB). It uses TensorFlow and Keras to build and train the model, with detailed visualizations and evaluation.

## Dataset

📂 **German Traffic Sign Recognition Benchmark (GTSRB)**

You can access and download the dataset from the following link:
[Download GTSRB Dataset](https://sid.erda.dk/public/archives/ff17dc924eba88d5d01a807357d6614c/published-archive.html)

## Project Status

✅ **Status**: `Completed!`

## Features

Here's what the project includes:

- **Data Loading and Preprocessing**
    - Read images and labels from the dataset
    - Normalize image pixel values
    - One-hot encoding of class labels

- **Model Building (CNN)**
    - Layers: Conv2D, MaxPool2D, Flatten, Dense, Dropout
    - Activation: ReLU, Softmax
    - Loss: Categorical Crossentropy
    - Optimizer: Adam

- **Model Training and Evaluation**
    - Split data into training and testing using `train_test_split`
    - Visualize accuracy and loss using matplotlib
    - Evaluate the model using classification report and accuracy score

- **Visualization**
    - Training progress graphs
    - Confusion matrix and prediction accuracy

## Technologies Used

This project is developed in Python using the following libraries:

```python
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPool2D
from sklearn.model_selection import train_test_split
import pickle
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, classification_report
```

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/Bayhaqieee/Traffic_Sign_Detection.git
```

2. Navigate to the project directory:
```bash
cd Traffic_Sign_Detection
```

3. Open the Jupyter Notebook file (`traffic_sign_detection.ipynb`) and run all cells sequentially.

4. Alternatively, you can run the program in sections using Jupyter Notebook blocks.

---

This project provides a practical approach to computer vision in real-world scenarios using traffic sign classification. Contributions and improvements are welcome!

**Author:** Bayhaqie

