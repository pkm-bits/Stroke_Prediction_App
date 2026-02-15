"""
Stroke Prediction - Streamlit Web Application
Interactive ML classification app with model comparison
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Stroke Prediction ML App",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_model_results():
    """Load pre-computed model results"""
    if os.path.exists('model/model_results.csv'):
        return pd.read_csv('model/model_results.csv', index_col=0)
    return None

@st.cache_resource
def load_model(model_name):
    """Load a trained model"""
    model_map = {
        'Logistic Regression': 'logistic_regression_model.pkl',
        'Decision Tree': 'decision_tree_model.pkl',
        'kNN': 'knn_model.pkl',
        'Naive Bayes': 'naive_bayes_model.pkl',
        'Random Forest (Ensemble)': 'random_forest_ensemble_model.pkl',
        'XGBoost (Ensemble)': 'xgboost_ensemble_model.pkl'
    }
    
    filepath = f'model/{model_map.get(model_name)}'
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    return None

@st.cache_resource
def load_scaler():
    """Load the feature scaler"""
    if os.path.exists('model/scaler.pkl'):
        with open('model/scaler.pkl', 'rb') as f:
            return pickle.load(f)
    return None

@st.cache_resource
def load_label_encoders():
    """Load label encoders"""
    if os.path.exists('model/label_encoders.pkl'):
        with open('model/label_encoders.pkl', 'rb') as f:
            return pickle.load(f)
    return None

def calculate_metrics_from_data(y_true, y_pred, y_pred_proba=None):
    """Calculate evaluation metrics"""
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

def plot_metrics_comparison(df_results):
    """Plot metrics comparison across models"""
    fig = go.Figure()
    
    metrics = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
    
    for metric in metrics:
        fig.add_trace(go.Bar(
            name=metric,
            x=df_results.index,
            y=df_results[metric],
            text=df_results[metric].round(3),
            textposition='auto',
        ))
    
    fig.update_layout(
        title="Model Performance Comparison Across All Metrics",
        xaxis_title="ML Model",
        yaxis_title="Score",
        barmode='group',
        height=500,
        showlegend=True
    )
    
    return fig

def plot_confusion_matrix(cm, model_name):
    """Plot confusion matrix"""
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=['No Stroke', 'Stroke'],
        y=['No Stroke', 'Stroke'],
        text_auto=True,
        color_continuous_scale='Blues',
        title=f'Confusion Matrix - {model_name}'
    )
    fig.update_layout(height=400)
    return fig

def preprocess_uploaded_data(df, label_encoders, scaler):
    """Preprocess uploaded test data"""
    # Drop id column if exists
    if 'id' in df.columns:
        df = df.drop('id', axis=1)
    
    # Handle missing values
    if 'bmi' in df.columns:
        df['bmi'].fillna(df['bmi'].median(), inplace=True)
    
    # Separate target if exists
    y_true = None
    if 'stroke' in df.columns:
        y_true = df['stroke']
        X = df.drop('stroke', axis=1)
    else:
        X = df.copy()
    
    # Encode categorical variables
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col in label_encoders:
            try:
                X[col] = label_encoders[col].transform(X[col].astype(str))
            except:
                X[col] = 0  # Handle unknown categories
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    return X_scaled, y_true

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🏥 Stroke Prediction ML Classification</h1>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.radio(
        "Select Page:",
        ["Model Overview", "Model Comparison", "Make Predictions"]
    )
    
    # Load resources
    df_results = load_model_results()
    scaler = load_scaler()
    label_encoders = load_label_encoders()
    
    # PAGE 1: Model Overview
    if page == "Model Overview":
        st.header("📋 Model Overview")
        
        st.markdown("""
        ### About This Project
        This application demonstrates **6 different machine learning classification models** 
        for predicting stroke risk based on patient health data.
        
        #### 🎯 Models Implemented:
        1. **Logistic Regression** - Linear probabilistic classifier
        2. **Decision Tree** - Tree-based rule classifier
        3. **K-Nearest Neighbors (kNN)** - Instance-based learning
        4. **Naive Bayes** - Probabilistic classifier based on Bayes' theorem
        5. **Random Forest** - Ensemble of decision trees
        6. **XGBoost** - Gradient boosting ensemble method
        
        #### 📊 Evaluation Metrics:
        - **Accuracy**: Overall correct predictions
        - **AUC**: Area Under ROC Curve
        - **Precision**: Positive prediction accuracy
        - **Recall**: True positive detection rate
        - **F1 Score**: Harmonic mean of precision and recall
        - **MCC**: Matthews Correlation Coefficient
        """)
        
        if df_results is not None:
            st.markdown("### 📈 Overall Results")
            st.dataframe(df_results.style.highlight_max(axis=0, color='lightgreen'), 
                        use_container_width=True)
            
            # Display best model
            st.markdown("### 🏆 Best Performing Models")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                best_acc = df_results['Accuracy'].idxmax()
                st.metric("Best Accuracy", best_acc, 
                         f"{df_results.loc[best_acc, 'Accuracy']:.4f}")
            
            with col2:
                best_auc = df_results['AUC'].idxmax()
                st.metric("Best AUC", best_auc,
                         f"{df_results.loc[best_auc, 'AUC']:.4f}")
            
            with col3:
                best_f1 = df_results['F1'].idxmax()
                st.metric("Best F1 Score", best_f1,
                         f"{df_results.loc[best_f1, 'F1']:.4f}")
    
    # PAGE 2: Model Comparison
    elif page == "Model Comparison":
        st.header("📊 Model Comparison")
        
        if df_results is not None:
            # Metrics comparison chart
            st.markdown("### Performance Comparison")
            fig = plot_metrics_comparison(df_results)
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed metrics table
            st.markdown("### Detailed Metrics Table")
            st.dataframe(
                df_results.style.background_gradient(cmap='RdYlGn', axis=0),
                use_container_width=True
            )
            
            # Model rankings
            st.markdown("### 🥇 Model Rankings by Metric")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Top 3 by Accuracy")
                top_acc = df_results.nlargest(3, 'Accuracy')['Accuracy']
                for i, (model, score) in enumerate(top_acc.items(), 1):
                    st.write(f"{i}. **{model}**: {score:.4f}")
            
            with col2:
                st.markdown("#### Top 3 by F1 Score")
                top_f1 = df_results.nlargest(3, 'F1')['F1']
                for i, (model, score) in enumerate(top_f1.items(), 1):
                    st.write(f"{i}. **{model}**: {score:.4f}")
        else:
            st.warning("Model results not found. Please train models first.")
    
    # PAGE 3: Make Predictions
    elif page == "Make Predictions":
        st.header("🔮 Make Predictions")
        
        # Model selection
        model_options = [
            'Logistic Regression',
            'Decision Tree',
            'kNN',
            'Naive Bayes',
            'Random Forest (Ensemble)',
            'XGBoost (Ensemble)'
        ]
        
        selected_model = st.selectbox(
            "Select Model:",
            model_options
        )
        
        # File upload
        st.markdown("### 📁 Upload Test Data")
        st.info("Upload a CSV file with test data. The file can include the 'stroke' column for evaluation.")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload your test dataset in CSV format"
        )
        
        if uploaded_file is not None:
            # Load data
            test_data = pd.read_csv(uploaded_file)
            
            st.markdown("### 📋 Uploaded Data Preview")
            st.dataframe(test_data.head(10), use_container_width=True)
            st.write(f"**Total samples:** {len(test_data)}")
            
            # Load model and preprocessors
            model = load_model(selected_model)
            
            if model is not None and scaler is not None and label_encoders is not None:
                # Preprocess data
                try:
                    X_test, y_true = preprocess_uploaded_data(test_data, label_encoders, scaler)
                    
                    # Make predictions
                    y_pred = model.predict(X_test)
                    
                    try:
                        y_pred_proba = model.predict_proba(X_test)[:, 1]
                    except:
                        y_pred_proba = None
                    
                    st.success(f"✅ Predictions completed using **{selected_model}**")
                    
                    # Show predictions
                    st.markdown("### 🎯 Predictions")
                    pred_df = pd.DataFrame({
                        'Prediction': y_pred,
                        'Predicted_Class': ['No Stroke' if p == 0 else 'Stroke' for p in y_pred]
                    })
                    
                    if y_pred_proba is not None:
                        pred_df['Stroke_Probability'] = y_pred_proba
                    
                    st.dataframe(pred_df, use_container_width=True)
                    
                    # Download predictions
                    csv = pred_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions",
                        data=csv,
                        file_name=f'predictions_{selected_model.replace(" ", "_").lower()}.csv',
                        mime='text/csv'
                    )
                    
                    # If ground truth available, show metrics
                    if y_true is not None:
                        st.markdown("### 📊 Evaluation Metrics")
                        
                        metrics = calculate_metrics_from_data(y_true, y_pred, y_pred_proba)
                        
                        # Display metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
                            st.metric("AUC", f"{metrics['AUC']:.4f}")
                        with col2:
                            st.metric("Precision", f"{metrics['Precision']:.4f}")
                            st.metric("Recall", f"{metrics['Recall']:.4f}")
                        with col3:
                            st.metric("F1 Score", f"{metrics['F1']:.4f}")
                            st.metric("MCC", f"{metrics['MCC']:.4f}")
                        
                        # Confusion matrix
                        st.markdown("### 🔲 Confusion Matrix")
                        cm = confusion_matrix(y_true, y_pred)
                        fig_cm = plot_confusion_matrix(cm, selected_model)
                        st.plotly_chart(fig_cm, use_container_width=True)
                        
                        # Classification report
                        st.markdown("### 📄 Classification Report")
                        report = classification_report(
                            y_true, y_pred,
                            target_names=['No Stroke', 'Stroke'],
                            output_dict=True
                        )
                        st.dataframe(pd.DataFrame(report).transpose(), 
                                   use_container_width=True)
                
                except Exception as e:
                    st.error(f"Error processing data: {str(e)}")
                    st.info("Please ensure your data has the correct format and columns.")
            else:
                st.error("Model or preprocessors not loaded. Please train models first.")
        else:
            st.info("👆 Please upload a CSV file to make predictions")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>Built with Streamlit | ML Classification Project</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
