# Breast Cancer Prediction using Machine Learning

## Purpose

This project aims to predict the presence of breast cancer using various machine learning algorithms. It leverages the Wisconsin Breast Cancer Diagnostic dataset, which contains features computed from digitized images of fine needle aspirates of breast mass.

## Approach

1. **Data Loading and Preparation:**
   - The `sklearn.datasets` module is used to load the breast cancer dataset.
   - A pandas DataFrame is created to store the data, with features and target variables.
   - Data is explored using `info()`, `columns`, `head()`, `tail()`, `describe()`, and `groupby()` to understand its structure and distribution.

2. **Data Splitting:**
   - The dataset is split into training and testing sets using `train_test_split` from `sklearn.model_selection`. This allows for model evaluation on unseen data.

3. **Model Training and Evaluation:**
   - Various classification algorithms are applied, including:
     - K-Nearest Neighbors (KNN)
     - Gaussian Naive Bayes
     - Support Vector Classifier (SVC)
     - Logistic Regression
     - Decision Tree
     - Random Forest
     - Multi-layer Perceptron (MLP)
   - Each model is trained on the training data and its performance is evaluated using accuracy as a metric.

4. **Scaling:**
   - Standard scaling is performed on the features using `StandardScaler` from `sklearn.preprocessing`. This ensures features have zero mean and unit variance, improving model performance.
   - Models are retrained and reevaluated with the scaled data.

5. **Comparison and Conclusion:**
   - Accuracy scores of different models are compared to identify the best-performing one.
   - GaussianNB initially showed the highest accuracy, but after scaling, other models may perform better.
