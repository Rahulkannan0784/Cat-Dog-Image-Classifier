![Cat-Dog Classifier](cat_dog_classifier.png){: width="400px" }


# Cat-Dog Image Classifier

A Convolutional Neural Network (CNN) project built with TensorFlow and Keras to classify images as cats or dogs. This repository serves as a practical exercise in deep learning.

## Overview

This project trains a CNN to distinguish between images of cats and dogs using a structured dataset split into training and test sets.

## Dataset

The dataset structure should be:
- `datasets/train_set/`
  - `cats/`: Cat images
  - `dogs/`: Dog images
- `datasets/test_set/`
  - `cats/`: Cat images for testing
  - `dogs/`: Dog images for testing

## Prerequisites

Ensure the following libraries are installed:
- TensorFlow
- Keras
- NumPy

```sh
pip install tensorflow keras numpy
```

```markdown
## Model Architecture
The CNN includes:
1. Convolutional Layers with ReLU activation
2. Max Pooling Layers
3. Flattening and Fully Connected Layers
4. Output Layer with sigmoid activation

```

## Usage
1. Prepare your dataset as described.
2. Run the provided Python script to train and evaluate the model.

## Results
After training, evaluate the model's performance using accuracy metrics. Adjust hyperparameters and dataset size to improve results as needed.

## Acknowledgments
- This project uses publicly available datasets for training and testing purposes.
- TensorFlow and Keras documentation provided valuable guidance for building and training the CNN.




