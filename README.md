# PyTorch Feedforward Neural Network Design: Development Concepts and Comparisons

## Steps 1 to 5

## 1. Feedforward Neural Network for Multi-Class Classification

### Core Technology

A **Feedforward Neural Network (FNN)** is the simplest form of artificial neural network where data flows in one direction—from the input layer, through hidden layers, to the output layer—without cycles or loops. For multi-class classification, the output layer typically contains as many neurons as there are classes, each producing a score for its class.

#### Example Structure

- Input Layer: Accepts fixed-length feature vectors.
- Embedding Layer: Converts integer-encoded tokens to dense vectors for text data.
- Hidden Layers: One or more layers with non-linear activations (e.g., ReLU).
- Output Layer: Uses `nn.Linear` with `num_classes` output units.


### Implementation

```python
class SentimentFFN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, ...):
        super().__init__()
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # MLP defined below...
```

**Usage:** For IMDB sentiment analysis, `num_classes=2` for binary classification, but the same logic applies for multi-class setups.

***

## 2. Adding Batch Normalization

### Core Technology

**Batch Normalization (BN)** normalizes the output of a layer to improve training stability and speed. BN helps mitigate issues of internal covariate shift, allowing deeper networks and higher learning rates.

#### Example Usage

- `nn.BatchNorm1d` is applied after each linear layer, before the activation function.


### Implementation

```python
if use_bn:
    layers.append(nn.BatchNorm1d(hidden_dim))
```

**Effect:** BN reduces sensitivity to initial weights and improves convergence.

***

## 3. ReLU6 and LeakyReLU Activations

### Core Technology

Activation functions introduce non-linearity, enabling the network to discover complex patterns.

- **ReLU:** Output = max(0, x)
- **ReLU6:** Clamps output to  range, helping prevent exploding activations.
- **LeakyReLU:** Allows small negative values (slope typically 0.01), reducing the likelihood of dead neurons.


### Implementation

```python
if activation=="relu": act_fn = nn.ReLU()
elif activation=="relu6": act_fn = nn.ReLU6()
elif activation=="leakyrelu": act_fn = nn.LeakyReLU(0.01)
```

**Usage:** Change the activation for different learning characteristics; LeakyReLU improves learning for sparse gradients, ReLU6 is used in mobile-efficient networks.

***

## 4. L2 Regularization (Weight Decay) without Dropout

### Core Technology

**L2 Regularization** penalizes large weight values to reduce model complexity and overfitting. Implemented as `weight_decay` in the optimizer, which adds the L2 penalty to the loss function.

### Implementation

```python
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
```

**Effect:** Improves generalization without randomly disabling neurons (as dropout would).

***

## 5. Two Hidden Layers with Softmax Output

### Core Technology

Increasing hidden layers allows the network to learn more abstract features.

- **Multiple Hidden Layers:** Increases expressive power.
- **Softmax Output:** Converts logits to probabilities, often used for multi-class classification.


### Implementation

```python
if hidden_layers == 2:
    layers.append(nn.Linear(h1, h2))
    ...
    layers.append(nn.Linear(h2, num_classes))
# Softmax is typically applied during evaluation:
F.softmax(logits, dim=1)
```

**Usage:** Two hidden layers with BatchNorm and ReLU improve representational capacity; softmax converts final logits to class probabilities.

# Conceptual Comparison Table

| Design Choice | Classification | BatchNorm | Activation | Regularization | Layers \& Output | Typical Impact |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| 1 Hidden Layer FNN | Binary/Multi | No | ReLU | None | 1 hidden, logits | Fast, baseline accuracy |
| + BatchNorm | Binary/Multi | Yes | ReLU | None | 1 hidden, logits | More stable, better convergence |
| LeakyReLU/ReLU6 | Binary/Multi | Yes/No | LeakyReLU/ReLU6 | None | 1 hidden, logits | Improved learning for sparse gradients or bounded activations |
| L2 Regularization | Binary/Multi | Yes/No | ReLU/Other | L2 | 1 hidden, logits | Less overfit, better generalization |
| Two Hidden Layers + Softmax | Binary/Multi | Yes | ReLU | None | 2 hidden, softmax | Improved feature abstraction, probability outputs |


***
# Usage Guidelines
- Install `requirements.txt`
- python main.py
- Terminal output will generate:
  - **Sample IMDB reviews**
  - **Individual training model by steps:**
  - ```
--- Training Model: L2_Regularization | Device: cpu ---
Epoch 01 | Train Loss: 0.6182 | Train Acc: 65.62% | Val Loss: 0.6207 | Val Acc: 66.74%

--- Training Model: Two_HidLayers_SoftMax | Device: cpu ---
Epoch 01 | Train Loss: 0.6153 | Train Acc: 65.78% | Val Loss: 0.6378 | Val Acc: 63.82%
Epoch 02 | Train Loss: 0.5267 | Train Acc: 73.72% | Val Loss: 0.5750 | Val Acc: 70.78%
```

  - **Summary Table** (3 Epoch)
```| Step                    | Final Train Accuracy (%) | Final Val Accuracy (%) | Final Train Loss | Final Val Loss |
| ----------------------- | ------------------------ | ---------------------- | ---------------- | -------------- |
| BasicFeedforward        | 78.04                    | 73.40                  | 0.461            | 0.545          |
| BatchNorm               | 79.20                    | 64.12                  | 0.445            | 0.750          |
| LeakyReLU\_Activation   | 78.06                    | 69.01                  | 0.458            | 0.614          |
| L2\_Regularization      | 77.49                    | 75.15                  | 0.471            | 0.508          |
| Two\_HidLayers\_SoftMax | 78.91                    | 73.89                  | 0.449            | 0.527          |
```
  - **Review Predictions** (From various Steps or Questions)
    ```
    - 13  This movie was a complete disappointment from ...              Negative  ...                   Negative               [0.97, 0.03]
    - 5      The movie was absolutely fantastic! I love it.              Positive  ...                   Positive               [0.05, 0.95]
```
# Summary

- **Feedforward Neural Networks in PyTorch** are flexible and powerful for both binary and multi-class classification.
- **Batch Normalization** and **activation function customization** significantly influence training behavior and model performance.
- **L2 regularization** is a practical way to improve generalization (without dropout).
- **Adding deeper layers and softmax output** enables richer feature learning and interpretable probability outputs.

Each design variant tackles different real-world model training challenges—stability, abstraction, generalization, and interpretability. Project demonstrates these by showing metrics, plots, and predictions for each configuration, making it a practical learning reference for PyTorch neural network design.
