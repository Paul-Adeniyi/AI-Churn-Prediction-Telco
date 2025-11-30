#  Customer Churn Prediction Using Machine Learning (Telco Dataset)

This project presents an end-to-end machine learning solution for predicting customer churn in the telecommunications industry. It includes data preprocessing, feature engineering, model development, evaluation and export of results for business intelligence reporting.

The purpose of this work is to demonstrate applied AI and data analytics expertise through a real-world business problem, forming part of a broader portfolio for professional development and the UK Global Talent Visa (Tech Nation-style) endorsement pathway.


# Project Overview

Customer churn refers to when an existing customer stops doing business with a company. For subscription-based companies such as telecoms providers, understanding and predicting churn is critical for revenue protection and customer retention.

In this project, I build a machine learning model that identifies customers who are likely to churn based on demographic attributes, billing information, and service usage patterns. The project follows a structured, industry-standard workflow:

- Data cleaning and transformation  
- One-hot encoding of categorical variables  
- Model training using Logistic Regression and Random Forest  
- Model evaluation with accuracy, classification report, confusion matrix, and ROC curve  
- Extraction of feature importance  
- Export of results for interactive Power BI reporting  
- Saving trained models for reproducibility and deployment


# Project Structure

AI-Churn-Prediction-Telco/

data
 - processed_telco_churn.csv
 - churn_model_results.csv

 models
 -logistic_regression_model.pkl
 -random_forest_model.pkl
 -scaler.pkl

 notebooks
  -data_preprocessing.ipynb
  -model_training_evaluation.ipynb

README.md
requirements.txt


# 1. Data Preprocessing

Data cleaning includes:

- Converting the `TotalCharges` column to numeric  
- Handling missing values using median imputation  
- Dropping unique identifiers (`customerID`)  
- One-hot encoding categorical variables  
- Standardising numerical features for logistic regression  
- Saving the processed dataset into `processed_telco_churn.csv`

This stage ensures the data is machine-learning ready and avoids inconsistencies or bias.


# 2. Model Development

Two models were implemented:

# Logistic Regression  
- Scaled training features using StandardScaler  
- Increased the number of iterations to ensure convergence  
- Provides interpretable linear relationships  

# Random Forest Classifier  
- Ensemble-based model that handles complex interactions  
- Identifies the most influential features contributing to churn  
- Achieved the strongest performance in this project  


# 3. Model Evaluation

The models were evaluated using standard metrics for classification:

- Accuracy  
- Precision, Recall, F1-score  
- Confusion Matrix
- ROC–AUC Score  
- Feature Importance

The ROC curve compares the probability predictions of both models.  
Feature importance reveals the drivers of churn, such as:

- Contract type  
- Tenure  
- Monthly charges  
- Internet service type  


# 4. Key Insights

Key business insights from the Random Forest model include:

- Customers with month-to-month contracts have a significantly higher likelihood of churning.  
- Long-term customers (high tenure) are less likely to leave the service.  
- Higher monthly charges correlate with higher churn risk.  
- Fibre optic customers show higher churn probability.  
- Payment method strongly impacts churn (e.g., electronic check users churn more).

These findings can directly support retention strategies and targeted intervention campaigns.



# 5. Power BI Dashboard

The file `churn_model_results.csv` was imported into Power BI to create:

- A churn probability distribution  
- Actual vs predicted churn charts  
- Breakdown by key attributes  
- Insights for stakeholder decision-making  
- Recommendations for reducing churn  

This helps bridge data science with business intelligence.


# 6. How to Reproduce the Project

#Clone the repository
git clone https://github.com/Paul-Adeniyi/AI-Churn-Prediction-Telco


# Install dependencies
pip install -r requirements.txt

# Run the notebooks
- data_preprocessing.ipynb
- model_training_evaluation.ipynb

# Power BI
Load the file:  
churn_model_results.csv

# Technologies Used

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  
- Google Colab  
- Joblib  
- Power BI  


# Dataset Source

Telco Customer Churn Dataset
https://www.kaggle.com/datasets/blastchar/telco-customer-churn


# Author

Paul Adeniyi Adesina 
MSc – Applied Artificial Intelligence & Data Analytics
Data Scientist/Analyst/ Aspiring Global Talent Visa Applicant  
GitHub: https://github.com/Paul-Adeniyi
LinkedIn: https://linkedin.com/in/paul-adeniyi-138b4b243


# Future Improvements

- Hyperparameter optimisation (GridSearchCV or Optuna)  
- Testing XGBoost, LightGBM, and Gradient Boosting models  
- Developing an interactive Streamlit web app  
- Deploying the model with FastAPI or Flask  
- Adding SHAP-based explainability  


# Project Significance

This project demonstrates:

- Ability to build end-to-end AI solutions  
- Strong understanding of business analytics  
- Practical data preprocessing skills  
- Model evaluation and interpretation  
- Effective communication of technical insights  
- Commitment to continuous learning  

It forms part of a growing portfolio supporting my application for the UK Global Talent Visa in Digital Technology.

Note: Model binaries (.pkl files) of scaler and random_forest_model are not included in this repository due to GitHub’s 25MB file size limit. 
However, they can be easily reproduced by running the provided notebook.





