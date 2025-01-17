# Customer Churn ANN Classification

## Live Demo:
[![Streamlit App](https://img.shields.io/badge/Streamlit-App-blue)](https://customer-churn-ann-classification-by-prince.streamlit.app/)

## Basic Steps

### 1. Forward and Backward Propagation

- **Forward Propagation**: Passes the input data through the network to generate the output.
- **Backward Propagation**: Computes the gradient of the loss function with respect to each weight by the chain rule, updating weights to minimize the loss.

### 2. Loss Functions

- **Binary Cross-Entropy**: Used for binary classification tasks, measures the difference between the predicted probabilities and the actual labels.

### 3. Optimizers

- **Adam Optimizer**: Combines the best properties of the AdaGrad and RMSProp algorithms to handle sparse gradients and non-stationary objectives.

### 4. Activation Functions

- **ReLU (Rectified Linear Unit)**: Introduces non-linearity into the model, helps the network learn complex patterns.
- **Sigmoid**: Squashes output to a range between 0 and 1, suitable for binary classification.

## Project Overview

This project classifies customer churn using an Artificial Neural Network (ANN). It takes input features about customers and predicts the probability of churn.

## Setup Instructions

### 1. Clone the Repository

To clone the repository, use the following command:

```bash
git clone https://github.com/prince2004patel/Customer-Churn-ANN-Classification.git
```

### 2. Install Dependencies:

1. Ensure you are in the project directory:
   
```bash
cd Customer-Churn-ANN-Classification
```

2. Install all necessary Python libraries:

```bash
pip install -r requirements.txt
```

### 3. Run The Application

Start the Streamlit app:

```bash
pip install -r requirements.txt
```
