# Machine Learning: Data Preprocessing & Regression Guide

## Part 1: Data Preprocessing

### 1. The Machine Learning Workflow
```mermaid
flowchart LR
    A[Dataset] --> B[Data Preprocessing]
    B --> C[Train-Test Split]
    C --> D[Model Training]
    D --> E[Prediction]
    E --> F[Evaluation]
```

* **Importing:** Dataset ingestion (e.g., `pandas.read_csv`). Creating the Feature Matrix ($X$) and the Target Vector ($y$).
* **Modeling:** Algorithm selection and training phase (fitting the model to $X_{train}$ and $y_{train}$).
* **Evaluating:** Inference on unseen data ($X_{test}$) and comparing predictions ($\hat{y}$) against actual results ($y_{test}$) using error metrics.

### 2. Train-Test Split Rationale

* **The Problem:** Overfitting. The model memorizes noise instead of learning the underlying signal.
* **The Solution:** Data partitioning.
  * **Train Set (~80%):** Used strictly for coefficient optimization and learning.
  * **Test Set (~20%):** Kept completely isolated. Used *only* for final performance validation.
### 3. Feature Scaling

Many Machine Learning algorithms rely on **Euclidean distance** to measure similarity between observations.

When features have vastly different scales, variables with larger magnitudes dominate the distance calculation.

#### Example Problem

| Age | Salary |
|------|---------|
| 20 | 20,000 |
| 30 | 50,000 |
| 40 | 90,000 |
| 50 | 150,000 |

Even though **Age** may be important, **Salary** contributes much more to distance calculations because its values are significantly larger.

```mermaid
flowchart LR
    A["Age<br/>20-50"] --> C["Distance Calculation"]
    B["Salary<br/>20k-150k"] --> C

    C --> D["Salary dominates"]
```

### Why Scale Features?

Feature scaling ensures that all variables contribute equally to the learning process.

```mermaid
flowchart LR
    A["Raw Features"] --> B["Feature Scaling"]
    B --> C["Equal Feature Contribution"]
    C --> D["Better Model Performance"]
```

---

### Normalization vs Standardization

| Feature | Normalization (Min-Max) | Standardization (Z-score) |
| :--- | :--- | :--- |
| **Formula** | $X_{norm} = \frac{X - X_{min}}{X_{max} - X_{min}}$ | $X_{stand} = \frac{X - \mu}{\sigma}$ |
| **Output Range** | [0, 1] | Typically [-3, 3] |
| **Mean** | Not preserved | Mean = 0 |
| **Standard Deviation** | Not preserved | Standard Deviation = 1 |
| **Best Use Case** | Deep Learning, bounded values | Most ML algorithms |
| **Outlier Sensitivity** | High | Lower |

---

### Normalization Example

```text
Original Values

Age:     20 30 40 50
Salary:  20000 50000 90000 150000
```

↓

```text
After Normalization

Age:     0.00 0.33 0.67 1.00
Salary:  0.00 0.23 0.54 1.00
```

```mermaid
flowchart LR
    A["Original Value"]
    --> B["Subtract Min"]
    --> C["Divide by Max - Min"]
    --> D["Value in Range [0,1]"]
```

---

### Standardization Example

```text
Original Values

Age:     20 30 40 50
Salary:  20000 50000 90000 150000
```

↓

```text
After Standardization

Age:    -1.2 -0.4 0.4 1.2
Salary: -1.3 -0.3 0.5 1.1
```

```mermaid
flowchart LR
    A["Original Value"]
    --> B["Subtract Mean"]
    --> C["Divide by Standard Deviation"]
    --> D["Mean = 0, Std = 1"]
```

---

## Quick Glossary (Key Terminology)
* **Matrix Creation:** In linear algebra, $X$ (Features) is a 2D table (rows = observations, columns = independent variables). $y$ (Target) is a 1D vector (a single column of labels to predict).
* **Inference:** The phase where a fully trained model makes predictions on brand-new, unseen data.
* **Error Metrics:** Mathematical formulas (like MSE or RMSE) calculating the distance between predictions ($\hat{y}$) and actual answers ($y$).
* **Overfitting:** When a complex model learns the dataset "by heart," performing perfectly on training data but failing on unseen data.
* **Euclidean Distance:** The straight-line distance between two points in multidimensional space.
* **Outliers:** Data points differing significantly from other observations.

---

## Part 2: Regression Models

### Section 4: Simple Linear Regression
```text
y
│
│         ●
│      ●
│   ●
│ ●
└──────────────── x
      Regression Line
```

* **Concept:** Predicting a continuous target based on a single feature.
* **Equation:** $y = b_0 + b_1x$
* **Under the Hood:** Uses the Ordinary Least Squares (OLS) method to find the line where the sum of squared residuals is at absolute minimum.
* **Scaling Required?** No. The coefficient automatically compensates for the scale.

### Section 5: Multiple Linear Regression
* **Concept:** Predicting the target using multiple features.
* **Equation:** $y = b_0 + b_1x_1 + b_2x_2 + ... + b_nx_n$
* **The Dummy Variable Trap:** When encoding categorical data, always omit one column to avoid Multicollinearity.
* **Feature Selection:** Use Backward Elimination (evaluating P-values) to remove statistically insignificant variables.

### Section 6: Polynomial Regression
```text
y
│             ●
│         ●
│      ●
│   ●
│ ●
│
└──────────────── x

Curved relationship
```
* **Concept:** Used for non-linear relationships. It fits a curve to the data points.
* **Equation:** $y = b_0 + b_1x_1 + b_2x_1^2 + ... + b_nx_1^n$
* **Why "Linear"?** The unknown parameters we solve for (coefficients $b$) are strictly linear, even though features are raised to a power.

### Section 7: Support Vector Regression (SVR)
```text
        Upper Margin
────────────────────────

      ●      ●

========================
     Regression Line
========================

  ●          ●

────────────────────────
        Lower Margin
```

* **Concept:** Fits the error within a specific threshold (the epsilon-insensitive tube). Data inside the tube is ignored (0 error).
* **Support Vectors:** Points outside the tube dictating its position.
* **Scaling Required?** **Yes (Mandatory).** SVR lacks built-in coefficients to scale down large numbers.

### Section 8: Decision Tree Regression
```mermaid
flowchart TD
    A[Age < 30?]
    A -->|Yes| B[Income < 50k?]
    A -->|No| C[Prediction = 200]
    B -->|Yes| D[Prediction = 100]
    B -->|No| E[Prediction = 150]
```

* **Concept:** Non-continuous model that splits data into segments (leaves) based on logical conditions. Predictions are averages of the terminal leaf.
* **Visual Output:** Looks like a staircase in 1D, not a continuous line.
* **Scaling Required?** No. Relies on logical splits, not Euclidean distances.

### Section 9: Random Forest Regression
```mermaid
flowchart LR
    A[Data] --> B[Tree 1]
    A --> C[Tree 2]
    A --> D[Tree 3]

    B --> E[Average Prediction]
    C --> E
    D --> E
```

* **Concept:** Ensemble Learning. Builds a "forest" of multiple Decision Trees and averages their predictions.
* **Advantage:** Drastically reduces the overfitting typical of single Decision Trees.
* **Scaling Required?** No.

---

## Part 3: Evaluating Model Performance

### 1. R-Squared ($R^2$) - Goodness of Fit
```mermaid
flowchart LR
    A[R² = 0] --> B[Poor Fit]
    C[R² = 0.5] --> D[Moderate Fit]
    E[R² = 1.0] --> F[Perfect Fit]
```

Measures what percentage of the variance in the dependent variable ($y$) is explained by the independent variables ($X$).
* **Equation:** $R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$
  * $SS_{res}$ (Residual Sum of Squares): $\sum(y_i - \hat{y}_i)^2$
  * $SS_{tot}$ (Total Sum of Squares): $\sum(y_i - \bar{y})^2$
* **Interpretation:** Range is 0 to 1. 1 means a perfect fit; 0 means the model simply guesses the average. A negative value means the model is worse than a flat average line.

> **The Fatal Flaw:** Adding *any* new independent variable will cause the standard $R^2$ to either stay the same or increase. It will **never** decrease, even if the new data is completely random. This naturally leads to overfitting.

### 2. Adjusted R-Squared ($Adj R^2$)
```mermaid
graph LR
    A[Add Feature] --> B[R² increases]
    A --> C[Adjusted R²]
    C --> D{Useful Feature?}
    D -->|Yes| E[Increase]
    D -->|No| F[Decrease]
```
Solves the fatal flaw of standard $R^2$ by mathematically penalizing the model for adding useless variables.
* **Equation:** $Adj R^2 = 1 - \frac{(1 - R^2)(n - 1)}{n - p - 1}$
  * $n$ = Number of data points (sample size)
  * $p$ = Number of independent variables (features)
* **Under the Hood:** As you add features ($p$ increases), the denominator $n - p - 1$ gets smaller. This drags the overall Adjusted $R^2$ score down unless the new feature significantly improves the model.

> **MLOps Key Takeaway:** When building Multiple Linear Regression models, **always** use Adjusted $R^2$ to evaluate and compare models. It proves whether a new feature actually adds predictive power or just creates noise.