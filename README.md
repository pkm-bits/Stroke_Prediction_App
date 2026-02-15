# Stroke Prediction - ML Classification Project

🔗 **Live App**: [Coming Soon - Will be updated after deployment]  
📂 **GitHub Repository**: [Coming Soon - Will be updated after deployment]

---

## Problem Statement

Stroke is a leading cause of death and disability worldwide. Early prediction and identification of high-risk individuals can significantly improve patient outcomes through timely intervention. This project aims to develop and compare multiple machine learning classification models to predict the likelihood of stroke occurrence based on various health and demographic factors.

**Objective**: Build and evaluate 6 different classification models to accurately predict stroke risk, enabling healthcare providers to identify at-risk patients and implement preventive measures.

---

## Dataset Description

### Source
**Dataset**: Healthcare Stroke Prediction Dataset  
**File**: `healthcare-dataset-stroke-data.csv`  
**Total Samples**: 5,110 patient records  

### Features

The dataset contains the following features:

| Feature | Type | Description |
|---------|------|-------------|
| **id** | Numerical | Unique identifier for each patient |
| **gender** | Categorical | Patient gender (Male/Female/Other) |
| **age** | Numerical | Patient age in years |
| **hypertension** | Binary | 0 = no hypertension, 1 = has hypertension |
| **heart_disease** | Binary | 0 = no heart disease, 1 = has heart disease |
| **ever_married** | Categorical | Yes/No |
| **work_type** | Categorical | Type of occupation (Private/Self-employed/Govt_job/children/Never_worked) |
| **Residence_type** | Categorical | Urban/Rural |
| **avg_glucose_level** | Numerical | Average blood glucose level |
| **bmi** | Numerical | Body Mass Index |
| **smoking_status** | Categorical | formerly smoked/never smoked/smokes/Unknown |
| **stroke** | Binary (Target) | 0 = no stroke, 1 = stroke occurred |

### Data Characteristics

- **Target Variable**: `stroke` (Binary classification: 0 = No Stroke, 1 = Stroke)
- **Class Distribution**: Highly imbalanced dataset
  - No Stroke (Class 0): ~95% of samples
  - Stroke (Class 1): ~5% of samples
- **Missing Values**: BMI column contains some missing values (handled via median imputation)
- **Preprocessing**: 
  - Categorical features encoded using Label Encoding
  - Numerical features scaled using StandardScaler
  - Train-test split: 80-20 ratio with stratification

---

## Models Used

### Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|--------------|----------|-----|-----------|--------|----|----|
| **Logistic Regression** | [0.9511] | [0.8377] | [0.0] | [0.0] | [0.0] | [0.0] |
| **Decision Tree** | [0.9286] | [0.691] | [0.0741] | [0.04] | [0.0519] | [0.0192] |
| **kNN** | [0.9481] | [0.614] | [0.2] | [0.02] | [0.0364] | [0.0491] |
| **Naive Bayes** | [0.8679] | [0.8033] | [0.1654] | [0.42] | [0.2373] | [0.2033] |
| **Random Forest (Ensemble)** | [0.9521] | [0.8256] | [1.0] | [0.02] | [0.0392] | [0.138] |
| **XGBoost (Ensemble)** | [0.9462] | [0.8001] | [0.3333] | [0.1] | [0.1538] | [0.1609] |

**Note**: Run `python model/train_models.py` to train all models and generate actual metrics.

---

### Model Performance Observations

| ML Model Name | Observation about model performance |
|--------------|-------------------------------------|
| **Logistic Regression** | Expected to provide a good baseline with interpretable coefficients. May struggle with non-linear relationships but works well with scaled features. Typically shows moderate performance on imbalanced datasets. Fast training and prediction times. |
| **Decision Tree** | Capable of capturing non-linear patterns and feature interactions without feature scaling. Prone to overfitting on imbalanced data. May show lower precision due to creating overly specific rules for minority class. Good for understanding feature importance. |
| **kNN** | Performance heavily dependent on the choice of k and distance metric. Sensitive to feature scaling and class imbalance. May show good recall but lower precision on minority class. Computationally expensive for large datasets. Requires careful hyperparameter tuning. |
| **Naive Bayes** | Based on probabilistic assumptions (feature independence). Generally fast and efficient even with limited data. May underperform if features are highly correlated. Often shows good balance between precision and recall. Works well with categorical features. |
| **Random Forest (Ensemble)** | Expected to be one of the top performers. Robust to overfitting through ensemble averaging. Handles feature interactions and non-linearity well. Less sensitive to class imbalance compared to single decision tree. Provides reliable feature importance rankings. May achieve highest AUC score. |
| **XGBoost (Ensemble)** | Anticipated to deliver the best overall performance through gradient boosting. Excellent at handling imbalanced datasets with proper parameter tuning. Captures complex patterns and interactions. May achieve highest F1 score and MCC. Requires more computational resources but delivers superior predictive power. |

---

## Evaluation Metrics Explained

1. **Accuracy**: Proportion of correct predictions (both positive and negative)
   - Formula: (TP + TN) / (TP + TN + FP + FN)
   
2. **AUC (Area Under ROC Curve)**: Measures the model's ability to distinguish between classes
   - Range: 0 to 1 (higher is better)
   - AUC > 0.9: Excellent, 0.8-0.9: Good, 0.7-0.8: Fair
   
3. **Precision**: Proportion of positive predictions that are actually correct
   - Formula: TP / (TP + FP)
   - Important when false positives are costly
   
4. **Recall (Sensitivity)**: Proportion of actual positives correctly identified
   - Formula: TP / (TP + FN)
   - Critical in medical diagnosis where missing cases is dangerous
   
5. **F1 Score**: Harmonic mean of Precision and Recall
   - Formula: 2 × (Precision × Recall) / (Precision + Recall)
   - Balances precision and recall, especially useful for imbalanced data
   
6. **MCC (Matthews Correlation Coefficient)**: Balanced measure even for imbalanced classes
   - Range: -1 to +1 (higher is better)
   - Considered one of the best metrics for binary classification

---

## Project Structure

```
stroke-prediction-app/
│
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation (this file)
│
├── data/
│   └── healthcare-dataset-stroke-data.csv    # Stroke prediction dataset
│
├── model/
│   ├── train_models.py             # Model training script
│   ├── logistic_regression_model.pkl         # Trained Logistic Regression
│   ├── decision_tree_model.pkl               # Trained Decision Tree
│   ├── knn_model.pkl                         # Trained KNN
│   ├── naive_bayes_model.pkl                 # Trained Naive Bayes
│   ├── random_forest_ensemble_model.pkl      # Trained Random Forest
│   ├── xgboost_ensemble_model.pkl            # Trained XGBoost
│   ├── scaler.pkl                            # Feature scaler
│   ├── label_encoders.pkl                    # Label encoders
│   └── model_results.csv                     # Model evaluation results
│
└── .github/
    └── copilot-instructions.md     # Project instructions
```

---

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Local Setup

1. **Clone the repository**
   ```bash
   git clone <your-github-repo-url>
   cd stroke-prediction-app
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the models**
   ```bash
   python model/train_models.py
   ```
   This will:
   - Load and preprocess the dataset
   - Train all 6 classification models
   - Generate evaluation metrics
   - Save trained models and results

4. **Run the Streamlit app locally**
   ```bash
   streamlit run app.py
   ```
   The app will open in your browser at `http://localhost:8501`

---

## Streamlit App Features

### 1. Model Overview Page
- Introduction to the project and models
- Overall results table with all metrics
- Best performing models by metric
- Color-coded performance indicators

### 2. Model Comparison Page
- Interactive bar charts comparing all models
- Side-by-side metric visualization
- Heatmap-style results table
- Model rankings by different metrics

### 3. Make Predictions Page
- **Model Selection**: Choose from 6 trained models
- **CSV Upload**: Upload test data (with or without labels)
- **Data Preview**: View uploaded data before prediction
- **Predictions Display**: See predictions with probabilities
- **Download Results**: Export predictions as CSV
- **Evaluation Metrics**: If labels provided, see confusion matrix and classification report

---

## Deployment on Streamlit Community Cloud

### Step-by-Step Deployment Guide

1. **Push code to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit - Stroke Prediction ML App"
   git branch -M main
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [streamlit.io/cloud](https://streamlit.io/cloud)
   - Sign in with your GitHub account
   - Click "New App"
   - Select your repository
   - Choose branch: `main`
   - Set main file path: `app.py`
   - Click "Deploy"

3. **Wait for deployment** (usually 2-5 minutes)

4. **Access your live app** at the provided Streamlit URL

### Important Notes for Deployment
- Ensure all model files are committed to GitHub (they're required for the app)
- The free tier has 1GB storage limit - consider this for model sizes
- For test data upload, use small CSV files due to free tier limitations

---

## Usage Instructions

### For Model Training
```bash
# Train all 6 models and generate metrics
python model/train_models.py
```

### For Web Application
```bash
# Run locally
streamlit run app.py

# Access at: http://localhost:8501
```

### For Making Predictions
1. Navigate to "Make Predictions" page
2. Select a model from the dropdown
3. Upload a CSV file with test data
4. View predictions and download results
5. If your CSV includes 'stroke' column, evaluation metrics will be displayed

---

## Key Technologies

- **Machine Learning**: scikit-learn, XGBoost
- **Web Framework**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Model Persistence**: Pickle

---

## Model Training Parameters

| Model | Key Parameters |
|-------|---------------|
| Logistic Regression | `max_iter=1000, random_state=42` |
| Decision Tree | `max_depth=10, random_state=42` |
| KNN | `n_neighbors=5` |
| Naive Bayes | `GaussianNB (default parameters)` |
| Random Forest | `n_estimators=100, max_depth=10, random_state=42` |
| XGBoost | `max_depth=6, eval_metric='logloss', random_state=42` |

---