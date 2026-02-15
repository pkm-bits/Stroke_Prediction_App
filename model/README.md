# Stroke Prediction Models

This directory contains all trained machine learning models for stroke prediction, along with Python modules for easy model loading and inference.

## Directory Structure

```
model/
├── *.pkl files                    # Saved trained models (DO NOT DELETE)
├── train_models.py               # Script to train all models
├── logistic_regression.py        # Logistic Regression predictor
├── decision_tree.py              # Decision Tree predictor
├── knn.py                        # K-Nearest Neighbors predictor
├── naive_bayes.py                # Naive Bayes predictor
├── random_forest.py              # Random Forest predictor
├── xgboost_model.py              # XGBoost predictor
├── __init__.py                   # Package initialization
├── model_results.csv             # Model evaluation results
└── README.md                     # This file
```

## Saved Model Files (.pkl)

The following pickle files contain trained models and preprocessing objects:

- `logistic_regression_model.pkl` - Trained Logistic Regression model
- `decision_tree_model.pkl` - Trained Decision Tree model
- `knn_model.pkl` - Trained K-Nearest Neighbors model
- `naive_bayes_model.pkl` - Trained Gaussian Naive Bayes model
- `random_forest_ensemble_model.pkl` - Trained Random Forest ensemble
- `xgboost_ensemble_model.pkl` - Trained XGBoost ensemble
- `scaler.pkl` - StandardScaler for feature scaling
- `label_encoders.pkl` - Label encoders for categorical features

**⚠️ Important: Do not delete or modify these .pkl files!**

## Using Individual Model Modules

Each model has its own Python module that provides a predictor class for easy loading and inference.

### Quick Start Example

```python
# Import the desired predictor
from model.random_forest import RandomForestPredictor

# Initialize the predictor (automatically loads the .pkl file)
predictor = RandomForestPredictor()

# Prepare patient data
patient = {
    'gender': 'Male',
    'age': 67,
    'hypertension': 0,
    'heart_disease': 1,
    'ever_married': 'Yes',
    'work_type': 'Private',
    'Residence_type': 'Urban',
    'avg_glucose_level': 228.69,
    'bmi': 36.6,
    'smoking_status': 'formerly smoked'
}

# Make prediction
prediction = predictor.predict(patient)
stroke_probability = predictor.get_stroke_probability(patient)

print(f"Prediction: {'Stroke' if prediction[0] == 1 else 'No Stroke'}")
print(f"Stroke Probability: {stroke_probability:.4f}")
```

### Available Predictors

#### 1. Logistic Regression
```python
from model.logistic_regression import LogisticRegressionPredictor
predictor = LogisticRegressionPredictor()
```

#### 2. Decision Tree
```python
from model.decision_tree import DecisionTreePredictor
predictor = DecisionTreePredictor()

# Get feature importance
importances = predictor.get_feature_importance()
```

#### 3. K-Nearest Neighbors
```python
from model.knn import KNNPredictor
predictor = KNNPredictor()

# Find nearest neighbors
distances, indices = predictor.kneighbors(patient, n_neighbors=5)
```

#### 4. Naive Bayes
```python
from model.naive_bayes import NaiveBayesPredictor
predictor = NaiveBayesPredictor()
```

#### 5. Random Forest
```python
from model.random_forest import RandomForestPredictor
predictor = RandomForestPredictor()

# Get feature importance
importances = predictor.get_feature_importance()

# Get individual tree predictions
tree_predictions = predictor.get_tree_predictions(patient)
```

#### 6. XGBoost
```python
from model.xgboost_model import XGBoostPredictor
predictor = XGBoostPredictor()

# Get feature importance
importances = predictor.get_feature_importance(importance_type='gain')

# Get number of boosting rounds
n_rounds = predictor.get_num_boosting_rounds()
```

## Common Methods

All predictor classes share these common methods:

- `predict(data)` - Make binary prediction (0 or 1)
- `predict_proba(data)` - Get probability estimates [prob_no_stroke, prob_stroke]
- `get_stroke_probability(data)` - Get probability of stroke (0-1)
- `preprocess_input(data)` - Preprocess and scale input data

## Input Data Format

Patient data should be provided as a dictionary with the following keys:

```python
{
    'gender': str,              # 'Male', 'Female', 'Other'
    'age': int/float,          # Age in years
    'hypertension': int,       # 0 or 1
    'heart_disease': int,      # 0 or 1
    'ever_married': str,       # 'Yes' or 'No'
    'work_type': str,          # 'Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'
    'Residence_type': str,     # 'Urban' or 'Rural'
    'avg_glucose_level': float,# Average glucose level
    'bmi': float,              # Body Mass Index
    'smoking_status': str      # 'formerly smoked', 'never smoked', 'smokes', 'Unknown'
}
```

## Training Models

To retrain all models from scratch:

```bash
cd /Users/pkarurma/stroke-prediction-app/model
python train_models.py
```

This will:
1. Load and preprocess the data
2. Train all 6 models
3. Evaluate and compare performance
4. Save all models and artifacts as .pkl files
5. Generate model_results.csv with evaluation metrics

## Model Performance

See `model_results.csv` for detailed performance metrics including:
- Accuracy
- AUC (Area Under ROC Curve)
- Precision
- Recall
- F1 Score
- MCC (Matthews Correlation Coefficient)

## Dependencies

Required packages:
```
pandas
numpy
scikit-learn
xgboost
pickle
```

## Notes

- All models use the same preprocessed data (scaled and encoded)
- The scaler and label encoders are shared across all models
- Models are trained with fixed random seed (42) for reproducibility
- Data is split 80/20 for training/testing with stratification
