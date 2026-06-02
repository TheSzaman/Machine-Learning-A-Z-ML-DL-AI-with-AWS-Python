### Part 1: Data Preprocessing (Technical Deep Dive & Glossary)

**1. The Machine Learning Workflow**
* **Importing:** Dataset ingestion (`pandas.read_csv`).
  * Matrix creation: Features ($X$) and Target ($y$).
* **Modeling:** Algorithm selection.
  * Training phase (fitting the model to $X_{train}$ and $y_{train}$).
* **Evaluating:**  Inference on $X_{test}$.
  * Comparing predicted $\hat{y}$ with actual $y_{test}$ using error metrics.

**2. Train-Test Split Rationale**
* **Problem:** Overfitting, model memorizes noise instead of learning signal.
* **Solution:** Data partitioning. 
  * **Train Set (~80%):** Used for coefficient optimization/learning.
  * **Test Set (~20%):** Kept completely isolated. Used *only* for final validation.

**3. Feature Scaling: Normalization vs. Standardization**
* **Why it's required:** Algorithms calculating Euclidean distance will heavily bias towards variables with larger magnitudes.

**A. Normalization (Min-Max Scaling)**
* **Formula:** $X_{norm} = \frac{X - X_{min}}{X_{max} - X_{min}}$
* **Output Range:** Strictly $[0, 1]$.
* **Best Use Case:** When the data distribution is strictly non-Gaussian, or in specific Deep Learning applications (e.g., image pixel values 0-255).
* **Vulnerability:** Extremely sensitive to outliers.

**B. Standardization (Z-score Normalization)**
* **Formula:** $X_{stand} = \frac{X - \mu}{\sigma}$ (where $\mu$ is mean, $\sigma$ is standard deviation).
* **Output Range:** Typically $[-3, 3]$, centered around a mean of $0$.
* **Best Use Case:** The default choice for most ML algorithms. Assumes data follows a Gaussian (bell-curve) distribution.
* **Advantage:** Highly robust to outliers compared to Normalization.

---

### Quick Glossary (Key Terminology)

* **Matrix Creation ($X$ vs $y$):** In linear algebra, $X$ (Matrix of Features) is a 2D table (rows = observations, columns = independent variables). $y$ (Dependent Variable) is a 1D vector (a single column of the answers/labels the model is trying to predict).
* **Inference:** The phase where a fully trained model is used to make predictions on brand-new, unseen data.
* **Error Metrics:** Mathematical formulas (like MSE or RMSE) used to calculate exactly how far off the model's predictions ($\hat{y}$) are from the real answers ($y$).
* **Overfitting:** When a model is too complex and learns the dataset "by heart" (including random noise). It performs perfectly on training data but fails completely on new data.
* **Euclidean Distance:** The straight-line distance between two data points in multidimensional space. (If salary is in thousands and age is in tens, salary mathematically dictates the entire distance—hence the need for scaling).
* **Outliers:** Data points that differ significantly from other observations (e.g., a billionaire's salary in a dataset of average workers).

# Part 2: Regression Models

##  Section 4: Simple Linear Regression
* **Concept:** Predicting a continuous dependent variable (target) based on a single independent variable (feature).
* **Equation:** $y=b_0+b_1x$
* **Under the Hood:** Uses the Ordinary Least Squares (OLS) method. It draws multiple lines and calculates the distance from each data point to the line (residuals). The goal is to find the line where the sum of squared residuals is the absolute minimum.
* **Feature Scaling:** Not required. The coefficient automatically compensates for the scale of the variable.

---

##  Section 5: Multiple Linear Regression
* **Concept:** Predicting the target using multiple independent variables.
* **Equation:** $y=b_0+b_1x_1+b_2x_2+...+b_nx_n$
* **The Dummy Variable Trap:** When encoding categorical data (e.g., locations) into 1s and 0s, you must always omit one column to avoid Multicollinearity. Keeping all columns means they perfectly predict each other, confusing the model.
* **Feature Selection (Backward Elimination):** Do not throw all variables into the model. Use the P-value (statistical significance) to remove features step-by-step (e.g., dropping features with P-value > 0.05) to optimize the model.

---

##  Section 6: Polynomial Regression
* **Concept:** Used when the data does not have a linear relationship (a straight line doesn't fit). It draws a curve.
* **Equation:** $y=b_0+b_1x_1+b_2x_1^2+...+b_nx_1^n$
* **Why "Linear"?** It is called linear because the coefficients (the unknown parameters we solve for) are strictly linear, even though the feature variables are raised to a power.
* **Implementation:** Transform the matrix of features using `PolynomialFeatures`, then feed it into the standard `LinearRegression` model.

---

##  Section 7: Support Vector Regression (SVR)
* **Concept:** Instead of minimizing the error to exactly zero, SVR fits the error within a specific threshold (the epsilon-insensitive tube). Data points inside this tube are ignored.
* **Support Vectors:** The points outside the tube that dictate how the tube is positioned.
* **Feature Scaling:** Mandatory. SVR does not have built-in coefficients to scale down large numbers. You must use `StandardScaler` on both features and the target variable.

---

##  Section 8: Decision Tree Regression
* **Concept:** A non-continuous, non-linear model. It splits the dataset into segments (leaves) based on logical conditions. The prediction for a new point is the average of the values in its terminal leaf.
* **Visual Output:** In a 1D dataset, the graph looks like a staircase rather than a continuous line or curve.
* **Feature Scaling:** Not required. Decision trees rely on logical splits, not Euclidean distances.
* **Use Case:** Very powerful for high-dimensional datasets, but performs poorly on single-feature datasets.

---
##  Section 9: Random Forest Regression
* **Concept:** Ensemble Learning. It builds a "forest" of multiple Decision Trees rather than relying on just one.
* **Process:** It picks a random subset of data points, builds a Decision Tree, and repeats this multiple times. For a new prediction, it calculates the average of the predictions from all the individual trees.
* **Advantage:** Drastically reduces the overfitting that commonly occurs with single Decision Trees.
* **Feature Scaling:** Not required, for the same reasons as a single Decision Tree.

#  Part 3: Evaluating Regression Models Performance

## Understanding R-squared (Goodness of Fit)
* **Concept:** $R^2$ (Coefficient of Determination) measures how well the regression line fits the actual data points. It tells you what percentage of the variance in the dependent variable ($y$) is explained by the independent variables ($X$).
* **Equation:** $R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$
  * $SS_{res}$ (Residual Sum of Squares): $\sum(y_i - \hat{y}_i)^2$. How far the data points are from your predicted regression line.
  * $SS_{tot}$ (Total Sum of Squares): $\sum(y_i - \bar{y})^2$. How far the data points are from a simple horizontal line representing the average of $y$.
* **Interpretation:** 
  * Values range from $0$ to $1$. 
  * $1$ means a perfect fit (no error). $0$ means the model is no better than just guessing the average.
  * *Negative value:* The model is actually worse than drawing a flat average line (usually happens if the model equation is completely wrong).
* **🚨 The Fatal Flaw:** Adding *any* new independent variable to a model will cause the standard $R^2$ to either stay the same or increase. It will **never** decrease, even if you add completely random, useless data (like predicting salary based on a person's shoe size). This leads to overfitting.

---

## Understanding Adjusted R-Squared
* **Concept:** Solves the fatal flaw of standard $R^2$. Adjusted $R^2$ mathematically penalizes the model for adding independent variables that do not actually improve its predictive power.
* **Equation:** $Adj R^2 = 1 - \frac{(1 - R^2)(n - 1)}{n - p - 1}$
  * $R^2$ = Standard R-squared value
  * $n$ = Number of data points (sample size)
  * $p$ = Number of independent variables (features)
* **Under the Hood (How the penalty works):** 
  * Look at the denominator: $n - p - 1$. 
  * As you add more variables, $p$ increases, making the denominator smaller. 
  * Dividing by a smaller number makes the whole subtracted fraction larger, which **drags the Adjusted $R^2$ down**.
  * The only way for Adjusted $R^2$ to increase is if the new variable improves the standard $R^2$ enough to overcome this mathematical penalty.
* **Key Takeaway for MLOps/Interviews:** When building Multiple Linear Regression models, **always** use Adjusted $R^2$ instead of standard $R^2$ to evaluate performance and compare models. It is the only reliable way to know if adding a new feature actually helps or just creates noise.