# Stroke Prediction Model

## Project Overview
This project developed a machine learning model to predict the likelihood of strokes by analyzing patient health data, including age, cholesterol levels, blood pressure, glucose levels, and smoking status. The most effective model used was a Support Vector Machine (SVM) with a radial basis function kernel, demonstrating high accuracy and precision. The project also utilized Random Forest for comparison, ultimately confirming the superiority of the SVM model.

## Objectives
1. Apply exploratory data analysis (EDA) and data preprocessing.
2. Develop an accurate and reliable stroke prediction tool using machine learning techniques.
3. Build a user interface for users to observe stroke predictions.

## Tools and Technologies
- **Machine Learning**: Support Vector Machine (SVM), Random Forest
- **Data Preprocessing**: `pandas`, `scikit-learn`, `SMOTE`
- **User Interface**: Custom-built UI for prediction input

## Data Collection and Preprocessing
- **Dataset**: Patient records from Kaggle's Stroke Prediction Dataset, including variables such as age, hypertension, heart disease status, smoking status, and Body Mass Index (BMI).
- **Preprocessing Steps**:
  - Removal of extreme outliers.
  - Imputation of missing values using mode.
  - Encoding of categorical variables using one-hot encoding.

## Methodology
- **Support Vector Machine (SVM)**:
  - Utilized for its effectiveness in binary classification.
  - Enhanced with a radial basis function (RBF) kernel for handling non-linear relationships.
  - Addressed class imbalance using SMOTE and RandomOverSampler.
  - Hyperparameter tuning performed using grid search.

- **Random Forest**:
  - Used as a comparison model.
  - Benefits include preventing overfitting and offering accurate predictions.
  - Managed class imbalance using SMOTE and RandomOverSampler.
  - Hyperparameter tuning via Grid-Based Cross Validation.

## Results and Comparison
- **SVM Performance**:
  - Accuracy: ~95%
  - High precision and recall, indicating a well-balanced model.
  - Low false positives and negatives, suggesting good generalization.

- **Random Forest Performance**:
  - Slightly higher accuracy in training but lower precision compared to SVM.
  - High recall, but potentially overfits the training data.

## User Interface
A simple UI was built to gather necessary information and predict stroke probability using the model. The UI includes fields and radio buttons for input.

## Conclusion
The project successfully developed an SVM-based model for stroke prediction, demonstrating high accuracy and potential for clinical application. Future work will focus on integrating the model into clinical workflows and exploring real-time analytics for preventive healthcare strategies.
