# 🧠 Gradient Descent: The Engine of Linear Regression

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Topics](https://img.shields.io/badge/Topic-Machine%20Learning%2C%20Optimization%2C%20Linear%20Algebra-brightgreen)]()

A comprehensive, visual, and intuitive exploration of the Gradient Descent algorithm, implemented from scratch to solve a Linear Regression problem.

## 📖 Table of Contents

1.  [Introduction](#-introduction)
2.  [The Intuition: What is Gradient Descent?](#-the-intuition-what-is-gradient-descent)
3.  [Mathematics of Linear Regression](#-mathematics-of-linear-regression)
4.  [The Cost Function: Sum of Squared Errors (SSE)](#-the-cost-function-sum-of-squared-errors-sse)
5.  [The Algorithm: How Gradient Descent Works](#-the-algorithm-how-gradient-descent-works)
6.  [Visualizing the Descent](#-visualizing-the-descent)
7.  [Project Structure](#-project-structure)
8.  [Implementation & Code Walkthrough](#-implementation--code-walkthrough)
9.  [Results & Analysis](#-results--analysis)
10. [Advanced Topics & Next Steps](#-advanced-topics--next-steps)
11. [How to Contribute](#-how-to-contribute)
12. [License](#-license)

---

## 🚀 Introduction

Gradient Descent is a first-order iterative optimization algorithm for finding the **local minimum** of a differentiable function. This project demonstrates the core concept by using Gradient Descent to find the best-fit line for a simple dataset.

## 🧩 The Intuition: What is Gradient Descent?

Imagine you are blindfolded on a hill and want to find the bottom of a valley. You feel the slope of the ground with your feet and take a small step in the direction where the ground descends the steepest. You repeat this process until you can't feel any slope—you've reached the bottom!

*   **You** are the **model parameters** (slope `m`, intercept `c`).
*   The **shape of the valley** is our **Cost Function, `L(m, c)`**.
*   **Feeling the slope** is calculating the **gradient** (`dL/dm`, `dL/dc`).
*   The **size of your step** is the **Learning Rate, `α`**.

## 📐 Mathematics of Linear Regression

**Model Prediction:**
$$\hat{y}_i = m \cdot x_i + c$$

Where:
*   $\hat{y}_i$ is the predicted value.
*   $m$ is the slope/weight.
*   $c$ is the y-intercept/bias.
*   $x_i$ is the input feature.

## 📉 The Cost Function: Sum of Squared Errors (SSE)

**Cost Function (L):**
$$L(m, c) = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = \sum_{i=1}^{n} (y_i - (m \cdot x_i + c))^2$$

**Our goal is to find the parameters `m` and `c` that minimize `L(m, c)`.**

## 🔁 The Algorithm: How Gradient Descent Works

1.  **Initialize:** Start with random values for `m` and `c` (often 0).
2.  **Compute Gradient:** Calculate the partial derivatives.
    $$\frac{\partial L}{\partial m} = -\frac{2}{n} \sum_{i=1}^{n} x_i (y_i - \hat{y}_i)$$
    $$\frac{\partial L}{\partial c} = -\frac{2}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)$$
3.  **Update Parameters:**
    $$m := m - \alpha \cdot \frac{\partial L}{\partial m}$$
    $$c := c - \alpha \cdot \frac{\partial L}{\partial c}$$
4.  **Repeat:** Steps 2 and 3 until convergence.

## 📊 Visualizing the Descent

The animation in the `results/` directory shows Gradient Descent in action.

## 📁 Project Structure
gradient-descent-linear-regression
```
/
│
├── data/
│   └── sample_data.csv          # Simple dataset (Salary vs. Happiness)
│
├── notebooks/
│   └── Gradient_Descent_from_Scratch.ipynb  # Jupyter notebook with full walkthrough
│
├── src/
│   ├── gradient_descent.py      # Core implementation of the algorithm
│   ├── linear_regression.py     # Linear regression model class
│   └── visualize.py             # Helper functions for plotting
│
├── results/
│   ├── loss_convergence.png     # Plot of cost vs. iterations
│   └── final_fit.png            # Plot of the final best-fit line
│
├── requirements.txt             # Python dependencies
└── README.md                    # This file


```
##  💻 Implementation & Code Walkthrough

### 1. Core Gradient Descent Function

```python
# src/gradient_descent.py
import numpy as np

def compute_gradient(X, y, m, c):
    n = len(y)
    y_pred = m * X + c
    error = y_pred - y
    dL_dm = (2/n) * np.sum(X * error)
    dL_dc = (2/n) * np.sum(error)
    return dL_dm, dL_dc

def gradient_descent(X, y, m_init=0, c_init=0, alpha=0.01, epochs=1000):
    m, c = m_init, c_init
    history = []

    for i in range(epochs):
        dL_dm, dL_dc = compute_gradient(X, y, m, c)
        m = m - alpha * dL_dm
        c = c - alpha * dL_dc
        cost = np.sum(((m * X + c) - y) ** 2)
        history.append((m, c, cost))

    return m, c, history

```
# Main script
import pandas as pd
import numpy as np
from src.gradient_descent import gradient_descent

# Load data
data = pd.read_csv('data/sample_data.csv')
X = data['Salary'].values
y = data['Happiness_Index'].values

# Normalize data
X = (X - np.mean(X)) / np.std(X)

# Run Gradient Descent
m_optimized, c_optimized, history = gradient_descent(X, y, alpha=0.1, epochs=100)

print(f"Optimized Slope (m): {m_optimized:.4f}")
print(f"Optimized Intercept (c): {c_optimized:.4f}")












## 💻 Tech Stack  

- **Language:** Python  
- **Libraries:** NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn  
- **Environment:** Jupyter Notebook  

---

## 🏆 About the Author  

👨‍💻 **Laxman Bhimrao Khedkar**  
🎓 Computer Engineering Graduate | Aspiring Data Analyst / Data Scientist  

📧 **Email:** [khedkarlaxman823@gmail.com](mailto:khedkarlaxman823@gmail.com)  
🔗 **Portfolio:** [beacons.ai/laxmankhedkar](https://beacons.ai/laxmankhedkar)  
💼 **LinkedIn:** [linkedin.com/in/laxman-khedkar](https://www.linkedin.com/in/laxman-khedkar)  
🐙 **GitHub:** [github.com/Laxman7744](https://github.com/Laxman7744)  


### 📜 License

This project is open-source and available under the MIT License.

