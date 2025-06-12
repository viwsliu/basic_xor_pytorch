# XOR Classification with PyTorch

This project implements a simple neural network to solve the classic XOR logic problem using PyTorch. It includes manual dataset generation, model training, and data visualization.

---

## What is the XOR Problem?

XOR (exclusive OR) is a binary logic operation that returns `1` only when the inputs differ:

| Input A | Input B | Output (A XOR B) |
|---------|---------|------------------|
|   0     |    0    |        0         |
|   0     |    1    |        1         |
|   1     |    0    |        1         |
|   1     |    1    |        0         |

This problem is not linearly separable, meaning it cannot be solved by a simple linear model. A neural network with at least one hidden layer is required to model its non-linearity.

---

## Approach

1. **Dataset Creation**: Generate a noisy XOR dataset by duplicating and perturbing the core 4 points, making it more realistic for training.
2. **Model Architecture**: Define a small feedforward neural network with one hidden layer using PyTorch.
3. **Training**: Use mean squared error (MSE) loss and stochastic gradient descent (SGD) to train the model.
4. **Evaluation**: Visualize the decision boundary and final predictions.

---

## File Overview

### `data.py`
- Defines the original XOR points.
- Implements the `create_dataset` function:
  - Repeats each data point with added noise.
  - Shuffles the dataset.
  - Splits into training and test sets.
  - Converts to PyTorch tensors.

### `model.py`
- Defines a simple fully connected neural network:
  - Input: 2 neurons
  - Hidden: customizable (typically 4–8)
  - Output: 1 neuron with Sigmoid activation
- Uses ReLU/Sigmoid activations for non-linearity.

### `train.py`
- Trains the model over a number of epochs.
- Prints training loss every N epochs.
- Plots the loss curve and saves it to the `training_details/` directory.

### `main.py`
- Entry point for the project:
  - Loads data.
  - Instantiates and trains the model.
  - Prints final XOR predictions.
  - Optionally visualizes decision boundaries.

### `plot.py`
- Provides additional plotting utilities for visualizing the dataset and predictions.

---

Install dependencies with:
```bash
pip install -r requirements.txt