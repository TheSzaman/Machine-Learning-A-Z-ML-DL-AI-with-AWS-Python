### Section 1: Data Preprocessing (Technical Deep Dive & Glossary)

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

### 📖 Quick Glossary (Key Terminology)

* **Matrix Creation ($X$ vs $y$):** In linear algebra, $X$ (Matrix of Features) is a 2D table (rows = observations, columns = independent variables). $y$ (Dependent Variable) is a 1D vector (a single column of the answers/labels the model is trying to predict).
* **Inference:** The phase where a fully trained model is used to make predictions on brand-new, unseen data.
* **Error Metrics:** Mathematical formulas (like MSE or RMSE) used to calculate exactly how far off the model's predictions ($\hat{y}$) are from the real answers ($y$).
* **Overfitting:** When a model is too complex and learns the dataset "by heart" (including random noise). It performs perfectly on training data but fails completely on new data.
* **Euclidean Distance:** The straight-line distance between two data points in multidimensional space. (If salary is in thousands and age is in tens, salary mathematically dictates the entire distance—hence the need for scaling).
* **Outliers:** Data points that differ significantly from other observations (e.g., a billionaire's salary in a dataset of average workers).