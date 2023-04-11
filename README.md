## Multi-Feature Linear Regression Example
[Boston housing data](https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data) is used for this project.
## Getting Started
run `cargo run --release` in the project's root directory for demo.
***
In this project, we use a portion of dataset to train our model, and then test it using the remainder data.

Each row has 14 data points, last of them is our y (target) value. After we train our model, we use our w and b values to make predictions, namely $\hat{y}$.

$$
f_{\vec{w},b}(\vec{x}) = \vec{w}  \cdot  \vec{x} + b = \hat{y}
$$

We also use Feature scaling and normalize our data.

$$
x'_i = \frac  {x_i - \mu}{max(x) - min(x)}
$$

## Derivatives for Gradient Descent
#### $w$:
$$
\frac\delta{\delta{w}}J(\vec{w},b) = \frac{1}{m}\sum_{i=1}^{m}(f(\vec x^{(i)})-y^{(i)})\vec x^{(i)}
$$
Where $J(\vec{w},b)$ is our cost function:  $$\sum_{i=1}^{m}(\hat{y}^{(i)} -{y}^{(i)})^2$$

#### $b$:

$$
\frac\delta{\delta{b}}J(\vec{w},b) = \frac{1}{m}\sum_{i=1}^{m}(f(\vec x^{(i)})-y^{(i)})
$$