"""
ML Assignment 2 - Streamlit Web Application
Classification Models for Wine Quality Prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="ML Classification Models",
    layout="wide"
)

# Title and description
st.title("Machine Learning Classification Models")
st.markdown("""
### Wine Quality Prediction using Multiple ML Models
This application demonstrates 6 different classification models trained on the Wine Quality dataset.
Upload your test data or use the built-in dataset to evaluate model performance.
""")

st.markdown("---")

# Sidebar
st.sidebar.header("Configuration")

# Model selection dropdown
model_options = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'K-Nearest Neighbor': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': GaussianNB(),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
}

selected_model = st.sidebar.selectbox(
    "Select Model",
    list(model_options.keys()),
    index=4  # Default to Random Forest
)

st.sidebar.markdown("---")

# File upload option
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload your CSV file (test data)",
    type=['csv'],
    help="Upload a CSV file with the same features as the Wine Quality dataset"
)

@st.cache_data
def load_default_data():
    """Load Wine Quality dataset"""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    df = pd.read_csv(url, sep=';')
    return df

@st.cache_data
def prepare_data(df):
    """Prepare data for modeling"""
    # Create binary target (quality >= 6 is good wine)
    df['target'] = (df['quality'] >= 6).astype(int)
    X = df.drop(['quality', 'target'], axis=1)
    y = df['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns.tolist()

def train_model(model, X_train, y_train):
    """Train the selected model"""
    model.fit(X_train, y_train)
    return model

def calculate_metrics(y_test, y_pred, y_prob):
    """Calculate all evaluation metrics"""
    metrics = {
        'Accuracy': round(accuracy_score(y_test, y_pred), 4),
        'AUC Score': round(roc_auc_score(y_test, y_prob), 4),
        'Precision': round(precision_score(y_test, y_pred, average='weighted'), 4),
        'Recall': round(recall_score(y_test, y_pred, average='weighted'), 4),
        'F1 Score': round(f1_score(y_test, y_pred, average='weighted'), 4),
        'MCC Score': round(matthews_corrcoef(y_test, y_pred), 4)
    }
    return metrics

def plot_confusion_matrix(y_test, y_pred, model_name):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Low Quality', 'High Quality'],
                yticklabels=['Low Quality', 'High Quality'])
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    return fig

def train_all_models(X_train, X_test, y_train, y_test):
    """Train all models and return results"""
    results = {}
    
    for name, model in model_options.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        results[name] = {
            'Accuracy': round(accuracy_score(y_test, y_pred), 4),
            'AUC': round(roc_auc_score(y_test, y_prob), 4),
            'Precision': round(precision_score(y_test, y_pred, average='weighted'), 4),
            'Recall': round(recall_score(y_test, y_pred, average='weighted'), 4),
            'F1': round(f1_score(y_test, y_pred, average='weighted'), 4),
            'MCC': round(matthews_corrcoef(y_test, y_pred), 4)
        }
    
    return results

# Main content
st.header("Dataset Information")

# Load data
if uploaded_file is not None:
    try:
        user_df = pd.read_csv(uploaded_file)
        st.success("Custom dataset uploaded successfully!")
        
        # Check if it has required columns
        required_cols = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                         'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                         'pH', 'sulphates', 'alcohol']
        
        if 'quality' in user_df.columns or 'target' in user_df.columns:
            df = user_df
        else:
            st.warning("Uploaded file should contain 'quality' or 'target' column.")
            df = None
    except Exception as e:
        st.error(f"Error reading file: {e}")
        df = None
else:
    df = None

if df is not None:
    st.write(f"**Dataset Shape:** {df.shape[0]} rows, {df.shape[1]} columns")
    st.write("**Features:**")
    feature_cols = [col for col in df.columns if col not in ['quality', 'target']]
    st.write(", ".join(feature_cols))
    
    # Show class distribution
    if 'quality' in df.columns:
        df['target'] = (df['quality'] >= 6).astype(int)
    st.write(f"**Class Distribution:** {df['target'].value_counts().to_dict()}")
    
    st.markdown("---")
    
    # Prepare data and train model
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_data(df)
    
    st.header(f"Selected Model: {selected_model}")

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Metrics", "Confusion Matrix", "All Models Comparison"])

    with tab1:
        # Train selected model
        model = model_options[selected_model]
        trained_model = train_model(model, X_train, y_train)
        
        # Predictions
        y_pred = trained_model.predict(X_test)
        y_prob = trained_model.predict_proba(X_test)[:, 1] if hasattr(trained_model, 'predict_proba') else y_pred
        
        # Calculate metrics
        metrics = calculate_metrics(y_test, y_pred, y_prob)
        
        st.subheader("Evaluation Metrics")
        
        # Display metrics in columns
        metric_cols = st.columns(6)
        for i, (metric_name, value) in enumerate(metrics.items()):
            with metric_cols[i]:
                st.metric(label=metric_name, value=f"{value:.4f}")
        
        st.markdown("---")
        
        # Classification Report
        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, target_names=['Low Quality', 'High Quality'], output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)

    with tab2:
        st.subheader(f"Confusion Matrix - {selected_model}")
        
        # Plot confusion matrix
        fig = plot_confusion_matrix(y_test, y_pred, selected_model)
        st.pyplot(fig)
        
        # Interpretation
        cm = confusion_matrix(y_test, y_pred)
        st.markdown(f"""
        **Interpretation:**
        - True Negatives (Low Quality correctly predicted): **{cm[0][0]}**
        - False Positives (Low Quality predicted as High): **{cm[0][1]}**
        - False Negatives (High Quality predicted as Low): **{cm[1][0]}**
        - True Positives (High Quality correctly predicted): **{cm[1][1]}**
        """)

    with tab3:
        st.subheader("All Models Comparison")
        
        # Train all models and get comparison
        with st.spinner("Training all models..."):
            all_results = train_all_models(X_train, X_test, y_train, y_test)
        
        # Create comparison table
        comparison_df = pd.DataFrame(all_results).T
        comparison_df.index.name = 'Model'
        
        st.dataframe(
            comparison_df.style.highlight_max(axis=0, color='lightgreen'),
            use_container_width=True
        )
        
        # Bar chart comparison
        st.subheader("Visual Comparison")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        comparison_df.plot(kind='bar', ax=ax, width=0.8)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)

else:
    st.info("Please upload a test data CSV file using the sidebar to view model metrics.")
