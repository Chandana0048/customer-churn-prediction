# customer-churn-prediction

# 📊 Customer Churn Prediction using scikit-learn

A machine learning project to predict customer churn in a telecommunications company. This project uses Python and scikit-learn to build, train, and evaluate classification models capable of identifying customers likely to leave the company.



## 📌 Project Overview

Customer churn is one of the most critical business problems in the telecom industry. The goal of this project is to develop a predictive model that accurately classifies whether a customer will churn or not based on various attributes such as services subscribed, contract type, tenure, and monthly charges.



## 🎯 Objectives

- Build a supervised classification model using scikit-learn
- Apply data cleaning, preprocessing, and feature encoding
- Train and evaluate models using accuracy, confusion matrix, and classification reports
- Perform cross-validation to ensure model robustness
- Compare Logistic Regression and Decision Tree models



## 🗂️ Dataset

- **Source:** [Telco Customer Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn)
- **Format:** CSV  
- **Records:** 7,043 customers  
- **Features:** 21 attributes including demographic, account, and service details


## 📚 Technologies Used

- Python  
- Pandas, NumPy  
- scikit-learn  
- Matplotlib, Seaborn  
- Google Colab / VS Code  



## 📈 Project Workflow

1. **Data Collection**  
   Load dataset and inspect initial structure.

2. **Exploratory Data Analysis (EDA)**  
   Visualize class distribution and inspect feature relationships.

3. **Data Cleaning**  
   Handle missing values and drop irrelevant columns.

4. **Feature Engineering**  
   Encode categorical variables using LabelEncoder and scale numerical features using StandardScaler.

5. **Model Building**  
   Train **Logistic Regression** and **Decision Tree Classifier** models.

6. **Model Evaluation**  
   Measure model performance using Accuracy, Confusion Matrix, and Classification Report.

7. **Cross-Validation**  
   Apply 5-fold cross-validation to assess model consistency.

8. **Result Summary**  
   Compare models and select the best-performing one based on accuracy and generalization.



## 🔍 Key Results

- **Logistic Regression Accuracy:** ~79.8%  
- **Cross-validation Accuracy:** ~79.85%  
- Improved convergence by adjusting `max_iter` parameter to 1000.



## 📝 Key Learnings

- End-to-end ML workflow: data preprocessing, model building, evaluation, and validation.
- Handling categorical features using Label Encoding.
- The importance of scaling and tuning hyperparameters for better model performance.
- Using cross-validation to prevent overfitting and assess model stability.



## 📌 Future Improvements

- Hyperparameter tuning using GridSearchCV.
- Try ensemble models like Random Forest and XGBoost.
- Add a basic Flask web application for model deployment.
- Improve feature engineering with domain-specific insights.


📬 Contact
Chandana K P
[LinkedIn](https://www.linkedin.com/in/chandana-puttanagappa)

 🚀 Acknowledgements

Dataset by [Kaggle: Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)
