# Loan Default Risk Prediction
# Project Abstract
This project presents a machine learning model designed to predict the likelihood of a customer defaulting on a loan. Using a synthetic dataset of 10,000 customer records, this model analyzes various financial and personal attributes to provide a reliable "default risk" score.

The workflow involved comprehensive Exploratory Data Analysis (EDA), robust feature engineering, and rigorous preprocessing using Scikit-Learn pipelines to handle outliers, scaling, and encoding. Several classification algorithms were trained and evaluated, including Logistic Regression, Decision Trees, Random Forest, SVM, and XGBoost.

After systematic hyperparameter tuning with GridSearchCV, the XGBoost Classifier emerged as the champion model, achieving a final accuracy of 96% on the unseen test set. This high-performing model demonstrates a strong potential to help financial institutions make more informed lending decisions and minimize potential losses from loan defaults.

# Table of Contents
Project Abstract

Workflow

Dataset

Data Preprocessing and Feature Engineering

Model Training and Performance

Technologies Used

How to Run This Project

Conclusion

# Workflow
The project follows a structured machine learning pipeline:

Data Loading & Inspection: The dataset is loaded, and an initial inspection is performed to understand its structure, data types, and check for missing values or duplicates.

Exploratory Data Analysis (EDA): In-depth analysis of data distributions, correlations between features, and categorical feature analysis to derive initial insights.

Data Preprocessing & Feature Engineering: The data is cleaned and prepared for modeling. This includes outlier handling, feature creation from existing columns (like signup_date), and encoding categorical variables.

Model Training & Hyperparameter Tuning: Several classification models are trained on the preprocessed data. GridSearchCV is used to find the optimal hyperparameters for each model to maximize performance.

Model Evaluation: The models are evaluated based on their accuracy, and a final champion model is selected.

Dataset
The model is trained on a synthetic dataset containing 10,000 customer records and 21 initial features, including:

Customer Demographics: age, num_dependents, marital_status, education

Financial Information: income, savings, monthly_expenses, credit_score, loan_amount

Behavioral Data: employment_years, has_credit_card, recent_default

Target Variable: target_default_risk (1 for default, 0 for non-default)

The target variable was found to be well-balanced, eliminating the need for techniques like SMOTE.

# Data Preprocessing and Feature Engineering
A robust preprocessing pipeline was built using Scikit-Learn to ensure consistency and reproducibility:

Outlier Handling: Outliers in numerical features like income and loan_amount were capped using the IQR method.

Feature Engineering: The signup_date column was engineered to extract the year, month, day, and day of the week.

Feature Dropping: customer_id and the original signup_date were dropped as they offer no predictive value.

Encoding:

Ordinal Encoding was applied to the education feature.

One-Hot Encoding was used for nominal features like home_ownership, marital_status, and region.

Imputation: Missing numerical values were imputed with the mean, and categorical values with the mode.

Scaling: All numerical features were scaled using StandardScaler.

# Model Training and Performance
The following models were trained and evaluated:

Logistic Regression

Decision Tree Classifier

Random Forest Classifier

Support Vector Machine (SVM)

XGBoost Classifier

After hyperparameter tuning, the XGBoost Classifier provided the best performance.

Final Model Test Accuracy: 96.0%

# Baseline Model Performance (Before Tuning)
This table shows the performance of each model using its default parameters.

     Models	                Accuracy	Precision	Recall	F1-Score

Decision Tree Classifier	    95.20%	 95.00%	  95.20%	 95.00%

Random Forest Classifier	    94.55%	 95.00%	  94.55%	 95.00%

Support Vector Machine (SVM)	93.70%	 94.00%	  93.70%	 94.00%

XGBoost Classifier	          96.00%	 96.00%	  96.00%	 96.00% 🏆

# Here are the performance metrics for the models after hyperparameter tuning.

     Models	                Accuracy	Precision	 Recall	  F1-Score
Decision Tree Classifier	    95.20%	 95.00%	   95.20%	   95.00%

Random Forest Classifier      94.55%	  95.00%	 94.55%	   95.00%

Support Vector Machine (SVM)	93.70%	  94.00%	  93.70%	  94.00%
 
#XGBoost Classifier	          96.00%	  96.00%	  96.00%	  96.00% 🏆



## How to Run This Project

To run this project on your local machine, please follow these steps:

**1. Clone the Repository**
```bash
git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
cd your-repository-name
```

**2. Create and Activate a Virtual Environment** (Recommended)
```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

**3. Install Required Libraries**
Make sure you have a `requirements.txt` file. Then, run:
```bash
pip install -r requirements.txt
```

**4. Launch Jupyter Notebook**
Run the following command in your terminal and open the main project file.
```bash
jupyter notebook "ml_project_the_loan_main (new).ipynb"
```

