# 📚 Part 2: Regression Models

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