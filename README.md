# Customer Churn Analysis

End-to-end data science project analyzing customer churn for a bank using the Churn Modelling dataset. Covers dataset evaluation, exploratory data analysis, and machine learning classification.

---

## Dataset

- 10,000 observations, 14 features
- Target variable: `Exited` (1 = churned, 0 = retained)
- Class distribution: ~20% churn, ~80% retained
- No missing values
- Features: CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary

---

## Project Structure

| Notebook | Description |
|---|---|
| `Customer_Churn_dataset_evaluation.ipynb` | Initial dataset inspection: shape, dtypes, descriptive statistics, null check |
| `Customer_Churn_EDA.ipynb` | Exploratory data analysis: visualizations and statistical tests |
| `Customer_Churn_ML.ipynb` | Machine learning: preprocessing, model training, evaluation, and churn risk predictions |

---

## EDA

- Correlation heatmap across all numeric features
- Bar charts: balance vs churn, age vs churn, geography vs balance, gender vs balance, churn distribution
- Welch's t-tests with Cohen's d on balance and age vs churn
- Crosstab analysis: HasCrCard and IsActiveMember vs churn
- Chi-square tests with Cramer's V: Gender, Geography, IsActiveMember, HasCrCard vs churn

**Key findings:**
- Higher-balance customers churn more — indicating unmet demand for wealth management services
- Older customers churn at a higher rate
- Inactive members are significantly more likely to churn (Cramer's V = 0.156)
- Gender is significantly associated with churn (Cramer's V = 0.106)
- Geography is significantly associated with churn
- Having a credit card has no significant association with churn (p = 0.492)

---

## Machine Learning

**Preprocessing:**
- Dropped non-predictive identifier columns (RowNumber, CustomerId, Surname)
- One-hot encoded Geography and Gender
- Stratified 80/20 train/test split
- StandardScaler applied to continuous features via ColumnTransformer pipeline
- Class imbalance handled via `class_weight="balanced"` and `scale_pos_weight`

**Models trained:**
- Logistic Regression
- Random Forest
- XGBoost
- CatBoost
- LightGBM

**Evaluation metrics:** Accuracy, ROC-AUC, precision, recall, F1-score, confusion matrix

**Output:** Top 20 highest churn-risk customers ranked by predicted churn probability

---

## Business Recommendations

**Target inactive members proactively.** Inactive members account for 63.9% of churned customers...

**Develop wealth management offerings for high-balance customers.** Churned customers hold...

**Prioritize retention efforts in Germany.** Geography is significantly associated with churn...

**Age-targeted retention for middle-aged customers.** Churned customers are on average older...

**Deprioritize credit card status as a churn signal.** HasCrCard showed no statistically significant association with churn (p = 0.492)...

---

## Tools & Libraries

Python, pandas, numpy, matplotlib, seaborn, scipy, scikit-learn, xgboost, catboost, lightgbm
