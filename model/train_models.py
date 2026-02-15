"""
Stroke Prediction - ML Classification Models Training
This script trains 6 classification models and evaluates them with comprehensive metrics
"""

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def load_and_preprocess_data(filepath):
    """Load and preprocess the stroke dataset"""
    print("Loading dataset...")
    df = pd.read_csv(filepath)
    
    print(f"Dataset shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nTarget distribution:\n{df['stroke'].value_counts()}")
    
    # Handle missing values
    if 'bmi' in df.columns:
        df['bmi'].fillna(df['bmi'].median(), inplace=True)
    
    # Drop id column if exists
    if 'id' in df.columns:
        df.drop('id', axis=1, inplace=True)
    
    # Encode categorical variables
    label_encoders = {}
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        if col != 'stroke':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    
    # Separate features and target
    X = df.drop('stroke', axis=1)
    y = df['stroke']
    
    print(f"\nFeatures shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    return X, y, label_encoders

def split_and_scale_data(X, y):
    """Split data into train/test and scale features"""
    print("\nSplitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    """Calculate all evaluation metrics"""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1': f1_score(y_true, y_pred, zero_division=0),
        'MCC': matthews_corrcoef(y_true, y_pred)
    }
    
    if y_pred_proba is not None:
        try:
            metrics['AUC'] = roc_auc_score(y_true, y_pred_proba)
        except:
            metrics['AUC'] = 0.0
    else:
        metrics['AUC'] = 0.0
    
    return metrics

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train all 6 models and evaluate them"""
    
    results = {}
    trained_models = {}
    
    # 1. Logistic Regression
    print("\n" + "="*60)
    print("Training Logistic Regression...")
    lr_model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    y_pred_proba_lr = lr_model.predict_proba(X_test)[:, 1]
    results['Logistic Regression'] = calculate_metrics(y_test, y_pred_lr, y_pred_proba_lr)
    trained_models['Logistic Regression'] = lr_model
    print("✓ Logistic Regression completed")
    
    # 2. Decision Tree
    print("\n" + "="*60)
    print("Training Decision Tree...")
    dt_model = DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=10)
    dt_model.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_test)
    y_pred_proba_dt = dt_model.predict_proba(X_test)[:, 1]
    results['Decision Tree'] = calculate_metrics(y_test, y_pred_dt, y_pred_proba_dt)
    trained_models['Decision Tree'] = dt_model
    print("✓ Decision Tree completed")
    
    # 3. K-Nearest Neighbors
    print("\n" + "="*60)
    print("Training K-Nearest Neighbors...")
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    y_pred_knn = knn_model.predict(X_test)
    y_pred_proba_knn = knn_model.predict_proba(X_test)[:, 1]
    results['kNN'] = calculate_metrics(y_test, y_pred_knn, y_pred_proba_knn)
    trained_models['kNN'] = knn_model
    print("✓ K-Nearest Neighbors completed")
    
    # 4. Naive Bayes
    print("\n" + "="*60)
    print("Training Naive Bayes...")
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    y_pred_nb = nb_model.predict(X_test)
    y_pred_proba_nb = nb_model.predict_proba(X_test)[:, 1]
    results['Naive Bayes'] = calculate_metrics(y_test, y_pred_nb, y_pred_proba_nb)
    trained_models['Naive Bayes'] = nb_model
    print("✓ Naive Bayes completed")
    
    # 5. Random Forest
    print("\n" + "="*60)
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, max_depth=10)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]
    results['Random Forest (Ensemble)'] = calculate_metrics(y_test, y_pred_rf, y_pred_proba_rf)
    trained_models['Random Forest (Ensemble)'] = rf_model
    print("✓ Random Forest completed")
    
    # 6. XGBoost
    print("\n" + "="*60)
    print("Training XGBoost...")
    xgb_model = XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss', max_depth=6)
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
    results['XGBoost (Ensemble)'] = calculate_metrics(y_test, y_pred_xgb, y_pred_proba_xgb)
    trained_models['XGBoost (Ensemble)'] = xgb_model
    print("✓ XGBoost completed")
    
    return results, trained_models

def display_results(results):
    """Display results in a formatted table"""
    print("\n" + "="*100)
    print("MODEL EVALUATION RESULTS")
    print("="*100)
    
    df_results = pd.DataFrame(results).T
    df_results = df_results.round(4)
    
    # Reorder columns
    col_order = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
    df_results = df_results[col_order]
    
    print(df_results.to_string())
    print("="*100)
    
    return df_results

def save_models_and_results(trained_models, scaler, df_results, label_encoders):
    """Save trained models, scaler, and results"""
    print("\nSaving models and results...")
    
    # Save each model
    for model_name, model in trained_models.items():
        filename = f"model/{model_name.replace(' ', '_').replace('(', '').replace(')', '').lower()}_model.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        print(f"✓ Saved {model_name}")
    
    # Save scaler
    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("✓ Saved scaler")
    
    # Save label encoders
    with open('model/label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)
    print("✓ Saved label encoders")
    
    # Save results
    df_results.to_csv('model/model_results.csv')
    print("✓ Saved results to CSV")
    
    print("\nAll models and artifacts saved successfully!")

def main():
    """Main execution function"""
    print("="*100)
    print("STROKE PREDICTION - ML CLASSIFICATION MODELS TRAINING")
    print("="*100)
    
    # Load and preprocess data
    X, y, label_encoders = load_and_preprocess_data('data/healthcare-dataset-stroke-data.csv')
    
    # Split and scale data
    X_train, X_test, y_train, y_test, scaler = split_and_scale_data(X, y)
    
    # Train and evaluate models
    results, trained_models = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # Display results
    df_results = display_results(results)
    
    # Save everything
    save_models_and_results(trained_models, scaler, df_results, label_encoders)
    
    print("\n" + "="*100)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*100)
    
    return df_results

if __name__ == "__main__":
    main()
