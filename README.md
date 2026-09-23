# Customer Churn Prediction Using Machine Learning

## Telecom Customer Churn Analysis & Prediction

This project analyzes historical telecom customer data to identify factors associated with customer churn and develops machine learning classification models to predict customers who are at higher risk of leaving the company.

The project follows an end-to-end machine learning workflow covering **data exploration, data cleaning, feature engineering, categorical encoding, leakage prevention, model training, evaluation, feature analysis, and business interpretation**.

---

## 1. Project Overview

Customer churn is a major challenge for telecom companies because losing existing customers can negatively affect recurring revenue and increase the cost of acquiring new customers.

The objective of this project is to use historical customer information to:

* Understand customer churn patterns
* Identify factors associated with churn
* Prepare customer data for machine learning
* Build multiple classification models
* Compare model performance
* Identify customers with higher predicted churn probability
* Generate actionable business insights for customer retention

Three machine learning models were evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest

After removing leakage-prone variables and comparing the models, **Logistic Regression was selected as the final model based on the reported evaluation results.**

---

# 2. Business Problem

Telecom companies need to identify customers who are likely to leave before the churn actually occurs.

A predictive churn model can help businesses move from a reactive approach to a proactive retention strategy.

Instead of treating every customer equally, the company can use predicted churn probabilities to prioritize customers who may require additional engagement.

### Business Question

> **Can historical customer information be used to identify customers who are at higher risk of churn and understand the factors associated with customer attrition?**

---

# 3. Project Objectives

The major objectives of this project are:

* Analyze the distribution of customer churn.
* Explore relationships between customer characteristics and churn.
* Clean and prepare the telecom customer dataset.
* Handle missing values and categorical variables.
* Remove irrelevant, high-cardinality, and leakage-prone features.
* Build multiple machine learning classification models.
* Evaluate models using multiple performance metrics.
* Analyze important features associated with churn.
* Generate business insights that can support customer-retention strategies.

---

# 4. Dataset

The project uses a telecom customer dataset stored as:

```text
data/raw/telco.csv
```

The original dataset contains:

* **7,043 customers**
* **50 columns**

The target variable is:

```text
Churn Label
```

The target contains two classes:

| Churn Label | Customers |
| ----------- | --------: |
| No          |     5,174 |
| Yes         |     1,869 |

This corresponds to approximately:

* **73.5% Non-Churn**
* **26.5% Churn**

Because the target classes are not evenly distributed, accuracy alone is not sufficient for evaluating the model.

---

# 5. Project Workflow

The project follows this general workflow:

```text
Raw Telecom Dataset
        ↓
Data Understanding
        ↓
Exploratory Data Analysis
        ↓
Data Cleaning
        ↓
Missing Value Handling
        ↓
Feature Selection
        ↓
Leakage Prevention
        ↓
Categorical Encoding
        ↓
Train-Test Split
        ↓
Feature Scaling
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Feature Interpretation
        ↓
Business Insights
```

---

# 6. Exploratory Data Analysis

The initial analysis included:

* Dataset shape
* Descriptive statistics
* Data types
* Missing-value analysis
* Duplicate-value analysis
* Unique-value analysis
* Churn distribution
* Numerical feature analysis
* Correlation analysis
* Categorical feature analysis

### Examples of visual analysis

The project examined relationships such as:

* Churn distribution
* Monthly charges vs. churn
* Gender vs. churn
* City-level churn patterns
* Correlations between numerical variables

The analysis helped identify potential relationships between customer characteristics and churn before model development.

---

# 7. Data Cleaning

Several preprocessing steps were performed before model training.

### 7.1 Removing Irrelevant and High-Cardinality Features

The following variables were removed:

```text
Customer ID
City
Zip Code
Latitude
Longitude
Quarter
Population
Country
State
```

### Reasons

**Customer ID**

An identifier does not provide meaningful predictive information about customer behavior.

**City and Zip Code**

These features have high cardinality and would create a very large number of categorical variables if directly encoded.

**Latitude and Longitude**

These geographic variables were excluded from the initial modeling approach.

**Quarter**

The dataset contained only one quarter value, so it did not provide useful variation.

**Population**

Population was excluded from the modeling dataset as it was not considered useful for the customer-level churn prediction objective.

---

# 8. Handling Missing Values

The analysis identified missing values in variables such as:

* Offer
* Internet Type

The missing values in `Internet Type` were investigated against `Internet Service`.

Customers without internet service were represented by missing Internet Type values.

Therefore, missing values were replaced with:

```text
No internet
```

Similarly, missing values in `Offer` were replaced with:

```text
No offer
```

After preprocessing, the notebook verified that there were no remaining missing values in the modeling dataset.

---

# 9. Data Leakage Prevention

One of the most important parts of this project was identifying and addressing **data leakage**.

Data leakage occurs when a machine learning model receives information that would not realistically be available at the time the prediction is supposed to be made.

The project identified several variables that could leak information about the churn outcome.

These included:

```text
Customer Status
Churn Score
Churn Category
Churn Reason
Satisfaction Score
```

For example:

* `Churn Reason` directly describes why a customer churned.
* `Churn Category` contains information related to the churn outcome.
* `Customer Status` can indicate whether the customer is already churned.
* `Churn Score` is itself related to churn prediction.
* `Satisfaction Score` produced an unusually strong relationship with churn and was therefore investigated as a potential leakage variable.

The notebook initially produced extremely high model performance while `Satisfaction Score` was present.

The project subsequently removed `Satisfaction Score` and other leakage-prone variables before reporting the final model performance.

This produced a more realistic evaluation of the model's ability to predict churn.

---

# 10. Target Variable

The target variable was:

```text
Churn Label
```

The target was explicitly mapped into binary values:

```text
No  → 0
Yes → 1
```

Therefore:

```text
0 = No Churn
1 = Churn
```

---

# 11. Categorical Encoding

Categorical variables were converted into numerical representations before model training.

### Binary Variables

Binary variables such as:

```text
Gender
Under 30
Senior Citizen
Married
Dependents
Referred a Friend
Phone Service
Multiple Lines
Internet Service
Online Security
Online Backup
Device Protection Plan
Premium Tech Support
Streaming TV
Streaming Movies
Streaming Music
Unlimited Data
Paperless Billing
```

were encoded using `LabelEncoder`.

### Multi-Class Variables

Categorical variables containing multiple categories, such as:

```text
Internet Type
Contract
Payment Method
```

were converted using:

```python
pd.get_dummies()
```

with:

```python
drop_first=True
```

This converted categorical information into machine-learning-compatible numerical features while avoiding redundant dummy variables.

---

# 12. Train-Test Split

The dataset was divided into training and testing sets using:

```python
train_test_split()
```

Configuration:

```text
Test Size: 20%
Random State: 32
Stratify: Yes
```

The use of stratification helps preserve the distribution of churn and non-churn customers between the training and testing datasets.

---

# 13. Feature Scaling

`StandardScaler` was used for feature scaling.

The scaler was fitted only on the training data:

```python
scaler.fit_transform(xx_train)
```

The same learned transformation was then applied to the test data:

```python
scaler.transform(xx_test)
```

This prevents information from the test set from influencing the scaling process.

Feature scaling is particularly relevant for Logistic Regression because its optimization can be affected by differences in feature magnitude.

---

# 14. Machine Learning Models

Three classification algorithms were evaluated.

## 14.1 Logistic Regression

The first model was:

```python
LogisticRegression(random_state=32)
```

Logistic Regression was used as a baseline classification model and also provided interpretable coefficients that could be analyzed to understand the direction and relative strength of associations between features and churn.

---

## 14.2 Decision Tree

The second model was:

```python
DecisionTreeClassifier(random_state=42)
```

The Decision Tree was trained to identify nonlinear decision rules separating churn and non-churn customers.

The initial tree was intentionally kept relatively simple before potential hyperparameter tuning.

Potential tuning parameters identified in the notebook include:

```text
max_depth
min_samples_split
min_samples_leaf
```

---

## 14.3 Random Forest

The third model was:

```python
RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
```

The Random Forest combines multiple decision trees to produce an ensemble prediction.

The project also examined Random Forest feature importance to understand which variables contributed most to the model's predictions.

---

# 15. Model Evaluation

The models were evaluated using several metrics:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Confusion Matrix
* ROC Curve

Using multiple metrics is important because the dataset contains substantially more non-churn customers than churn customers.

---

# 16. Final Model Performance

After removing leakage-prone variables, the notebook reports the following model comparison:

| Model               | Accuracy | ROC-AUC | Churn Recall | Churn F1 |
| ------------------- | -------: | ------: | -----------: | -------: |
| Logistic Regression |   83.53% |  0.9002 |       67.38% |   0.6855 |
| Decision Tree       |   78.00% |  0.7341 |       64.00% |   0.6100 |
| Random Forest       |   83.32% |  0.8894 |       61.00% |   0.6600 |

### Logistic Regression — Final Reported Metrics

The final Logistic Regression model achieved approximately:

```text
Accuracy   : 83.53%
ROC-AUC    : 0.9002
Churn Recall: 67.38%
Churn F1    : 0.6855
```

The model was selected as the final model in the project.

---

# 17. Why ROC-AUC Was Used

ROC-AUC evaluates how well the model separates churners from non-churners across different classification thresholds.

This is useful because the business may not always want to use the default:

```text
0.50
```

probability threshold.

For example, a company could potentially choose a different threshold depending on the relative cost of:

* Missing a customer who eventually churns
* Contacting a customer who would not have churned

The notebook also generated a ROC curve for the final Logistic Regression model.

---

# 18. Confusion Matrix

A confusion matrix was generated for the final Logistic Regression model.

It provides four types of outcomes:

```text
True Negative
False Positive
False Negative
True Positive
```

For churn prediction, correctly identifying actual churners is particularly important because these customers can potentially be targeted with retention actions.

---

# 19. Feature Interpretation

The Logistic Regression coefficients were analyzed to understand which features had stronger associations with the predicted churn outcome.

The notebook generated a coefficient table containing:

```text
Feature
Coefficient
Absolute_Coefficient
```

The coefficients were sorted using the absolute coefficient value to identify features with stronger model associations.

### Important interpretation

A positive coefficient indicates an association with higher predicted churn probability, while a negative coefficient indicates an association with lower predicted churn probability, holding the other model features constant.

These relationships should **not be interpreted as proof of causation**.

---

# 20. Key Business Insights

Based on the final analysis, several factors were associated with customer churn.

### 1. Monthly Charges

Higher monthly charges were associated with higher churn probability.

**Potential business action:**

Review pricing and perceived value among high-charge customers and consider appropriate retention offers or bundled services.

---

### 2. Customer Tenure

Longer tenure was associated with lower churn.

**Potential business action:**

Focus additional onboarding and engagement efforts on newer customers, particularly during the early stages of their relationship.

---

### 3. Contract Duration

Longer-term contracts were associated with lower churn compared with shorter contract arrangements.

**Potential business action:**

Consider suitable incentives and additional value for customers who move toward longer-term contracts.

---

### 4. Customer Referrals

Higher referral activity was associated with lower churn.

**Potential business action:**

Strengthen referral and loyalty programs to encourage customer engagement.

---

### 5. Dependents

Customers with dependents showed an association with lower churn.

**Potential business action:**

Explore family-oriented plans, shared services, and household benefits as part of customer-retention strategies.

---

### 6. Online Security and Technical Support

Services such as Online Security and Premium Tech Support showed negative associations with churn in the Logistic Regression model.

**Potential business action:**

Evaluate whether additional support and security services can be incorporated into appropriate retention or value packages.

---

### 7. High-Risk Customer Identification

The model generates churn probabilities that can be used to identify customers with relatively higher predicted churn risk.

This allows a company to prioritize:

* Proactive customer support
* Personalized offers
* Retention campaigns
* Customer engagement initiatives

rather than applying the same strategy to every customer.

---

# 21. Overall Business Recommendation

The analysis suggests that customer-retention efforts could particularly focus on customers with combinations of characteristics associated with higher predicted churn risk, such as:

```text
Higher monthly charges
+
Shorter tenure
+
Short-term contracts
+
Lower engagement indicators
```

The model can be used as a decision-support tool to prioritize customers for retention activities.

However, model predictions should be combined with business context rather than being treated as definitive explanations of customer behavior.

---

# 22. Project Limitations

The project has several limitations:

* The model is based on historical customer data.
* Historical relationships may not perfectly represent future customer behavior.
* Model performance depends on the quality of the available customer information.
* The identified relationships represent statistical associations rather than causal relationships.
* Several variables were removed because they could introduce data leakage or would not realistically be available before prediction.
* The current model uses a standard classification threshold.
* The model could be further optimized according to the business cost of false positives and false negatives.
* Hyperparameter tuning and production-grade pipelines could be developed further.

---

# 23. Future Improvements

Potential improvements include:

### Hyperparameter Optimization

Apply techniques such as:

```text
GridSearchCV
RandomizedSearchCV
```

to systematically optimize model parameters.

### Threshold Optimization

Instead of automatically using a 0.50 probability threshold, select a threshold based on the business cost of different prediction errors.

### Cross-Validation

Use stratified cross-validation to obtain a more robust estimate of model performance.

### Model Explainability

Add tools such as:

```text
SHAP
LIME
```

for more detailed customer-level explanations.

### Production Pipeline

Create a complete preprocessing and prediction pipeline so that new customer data can be passed through the same transformations used during training.

### Deployment

The final model could potentially be deployed through:

```text
Streamlit
Flask
FastAPI
```

and integrated into a customer-retention dashboard.

---

# 24. Technologies Used

### Programming Language

* Python

### Data Analysis

* Pandas
* NumPy

### Data Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn

### Models

* Logistic Regression
* Decision Tree
* Random Forest

### Evaluation

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Confusion Matrix
* ROC Curve

---

# 25. Project Structure

The project follows a structured organization similar to:

```text
customer_churn_project/
│
├── data/
│   └── raw/
│       └── telco.csv
│
├── notebooks/
│   └── customer_churn.ipynb
│
├── models/
│
├── outputs/
│   ├── logistic_regression_coefficients.csv
│   └── ...
│
└── README.md
```

The exact contents of `models/` and `outputs/` depend on the artifacts saved from the project workflow.

---

# 26. How to Run the Project

## 1. Clone the repository

```bash
git clone <repository-url>
```

## 2. Navigate into the project

```bash
cd customer_churn_project
```

## 3. Create a virtual environment

```bash
python -m venv pjt_env
```

## 4. Activate the environment

### Windows

```bash
pjt_env\Scripts\activate
```

### Linux / macOS

```bash
source pjt_env/bin/activate
```

## 5. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

## 6. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
notebooks/customer_churn.ipynb
```

Make sure the dataset is available at:

```text
data/raw/telco.csv
```

---

# 27. Project Artifacts

The notebook generates a Logistic Regression coefficient file:

```text
logistic_regression_coefficients.csv
```

This file contains:

```text
Feature
Coefficient
Absolute_Coefficient
```

It can be used for further feature interpretation and reporting.

---

# 28. Conclusion

This project demonstrates an end-to-end customer churn prediction workflow using telecom customer data.

The analysis progressed from exploratory data analysis and preprocessing to leakage detection, feature engineering, model training, evaluation, and business interpretation.

Three classification models were compared:

```text
Logistic Regression
Decision Tree
Random Forest
```

After removing leakage-prone variables, Logistic Regression achieved:

```text
83.53% Accuracy
0.9002 ROC-AUC
67.38% Churn Recall
0.6855 Churn F1
```

The project demonstrates how machine learning can be used to identify customers at higher risk of churn while also highlighting the importance of **data leakage prevention, appropriate evaluation metrics, and business interpretation**.

The model is intended as a **decision-support tool for customer-retention analysis**, rather than as a standalone replacement for business judgment.

---

## Author

**Jahid Khan**

Machine Learning / Data Analytics Project

---

## Disclaimer

This project is intended for educational, analytical, and portfolio purposes. The model's predictions represent statistical patterns in the available historical data and should not be interpreted as causal conclusions or guaranteed predictions of individual customer behavior.
