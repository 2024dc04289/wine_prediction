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
                             confusion_matrix)
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
### Wine Quality Prediction using 6 ML Models
This application evaluates 6 different classification models on the Wine Quality dataset.
Upload your test data CSV file to view model performance metrics.
""")

st.markdown("---")

# Sidebar - File upload only
st.sidebar.header("Upload Test Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload your CSV file",
    type=['csv'],
    help="Upload a CSV file with Wine Quality dataset format (must include 'quality' column)"
)

# Define all 6 models
model_options = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'K-Nearest Neighbor': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes (Gaussian)': GaussianNB(),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
}

@st.cache_data
def prepare_data(df):
    """Prepare data for modeling"""
    df = df.copy()
    df['target'] = (df['quality'] >= 6).astype(int)
    X = df.drop(['quality', 'target'], axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns.tolist()

def train_all_models(X_train, X_test, y_train, y_test):
    """Train all 6 models and return results with predictions"""
    results = {}
    predictions = {}
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'K-Nearest Neighbor': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes (Gaussian)': GaussianNB(),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        results[name] = {
            'Accuracy': round(accuracy_score(y_test, y_pred), 4),
            'AUC Score': round(roc_auc_score(y_test, y_prob), 4),
            'Precision': round(precision_score(y_test, y_pred, average='weighted'), 4),
            'Recall': round(recall_score(y_test, y_pred, average='weighted'), 4),
            'F1 Score': round(f1_score(y_test, y_pred, average='weighted'), 4),
            'MCC Score': round(matthews_corrcoef(y_test, y_pred), 4)
        }
        predictions[name] = {'y_pred': y_pred, 'y_prob': y_prob}
    
    return results, predictions

def plot_confusion_matrix(y_test, y_pred, model_name):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Low', 'High'],
                yticklabels=['Low', 'High'])
    ax.set_xlabel('Predicted', fontsize=9)
    ax.set_ylabel('Actual', fontsize=9)
    ax.set_title(f'{model_name}', fontsize=10, fontweight='bold')
    return fig

def get_model_observations(results):
    """Generate observations based on model performance"""
    df = pd.DataFrame(results).T
    
    best_accuracy = df['Accuracy'].idxmax()
    best_auc = df['AUC Score'].idxmax()
    best_f1 = df['F1 Score'].idxmax()
    best_mcc = df['MCC Score'].idxmax()
    
    observations = f"""
### Model Performance Observations

Based on the evaluation metrics calculated for all 6 classification models:

**1. Best Performing Models:**
- **{best_accuracy}** achieved the highest Accuracy of **{df.loc[best_accuracy, 'Accuracy']:.4f}**
- **{best_auc}** achieved the highest AUC Score of **{df.loc[best_auc, 'AUC Score']:.4f}**
- **{best_f1}** achieved the highest F1 Score of **{df.loc[best_f1, 'F1 Score']:.4f}**
- **{best_mcc}** achieved the highest MCC Score of **{df.loc[best_mcc, 'MCC Score']:.4f}**

**2. Individual Model Analysis:**

| Model | Observations |
|-------|-------------|
| **Logistic Regression** | Linear model, Accuracy={df.loc['Logistic Regression', 'Accuracy']:.4f}. Works well for linearly separable data. Provides interpretable coefficients. |
| **Decision Tree** | Non-linear model, Accuracy={df.loc['Decision Tree', 'Accuracy']:.4f}. Prone to overfitting but highly interpretable. |
| **K-Nearest Neighbor** | Instance-based learning, Accuracy={df.loc['K-Nearest Neighbor', 'Accuracy']:.4f}. Sensitive to feature scaling. |
| **Naive Bayes (Gaussian)** | Probabilistic model, Accuracy={df.loc['Naive Bayes (Gaussian)', 'Accuracy']:.4f}. Assumes feature independence, fast training. |
| **Random Forest** | Ensemble method, Accuracy={df.loc['Random Forest', 'Accuracy']:.4f}. Reduces overfitting through bagging. |
| **XGBoost** | Gradient boosting, Accuracy={df.loc['XGBoost', 'Accuracy']:.4f}. Excellent for structured/tabular data. |

**3. Key Insights:**
- Ensemble methods (Random Forest, XGBoost) generally provide more robust predictions
- MCC Score is particularly useful for imbalanced datasets as it considers all confusion matrix quadrants
- AUC Score measures the model's ability to distinguish between classes regardless of threshold
"""
    return observations

# Main content
st.header("Dataset Information")

# Load data
if uploaded_file is not None:
    try:
        # Try semicolon separator first (Wine Quality format), then comma
        try:
            user_df = pd.read_csv(uploaded_file, sep=';')
            if len(user_df.columns) == 1:
                uploaded_file.seek(0)
                user_df = pd.read_csv(uploaded_file, sep=',')
        except:
            uploaded_file.seek(0)
            user_df = pd.read_csv(uploaded_file)
        
        st.success("Dataset uploaded successfully!")
        
        if 'quality' in user_df.columns:
            df = user_df
        else:
            st.error("Uploaded file must contain 'quality' column.")
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
    df_display = df.copy()
    df_display['target'] = (df_display['quality'] >= 6).astype(int)
    class_dist = df_display['target'].value_counts().to_dict()
    st.write(f"**Class Distribution:** Low Quality (0): {class_dist.get(0, 0)}, High Quality (1): {class_dist.get(1, 0)}")
    
    st.markdown("---")
    
    # Prepare data and train all models
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_data(df)
    
    st.header("All 6 Models - Evaluation Results")
    
    with st.spinner("Training all 6 models..."):
        all_results, all_predictions = train_all_models(X_train, X_test, y_train, y_test)
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Comparison Table", "Confusion Matrices", "Observations"])
    
    with tab1:
        st.subheader("Evaluation Metrics Comparison Table")
        
        # Create comparison table
        comparison_df = pd.DataFrame(all_results).T
        comparison_df.index.name = 'Model'
        
        st.dataframe(
            comparison_df.style.highlight_max(axis=0, color='lightgreen').format("{:.4f}"),
            use_container_width=True
        )
        
        st.markdown("*Green highlighting indicates the best score for each metric*")
        
        # Bar chart comparison
        st.subheader("Visual Comparison")
        
        fig, ax = plt.subplots(figsize=(14, 6))
        comparison_df.plot(kind='bar', ax=ax, width=0.8)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Model Performance Comparison - All 6 Metrics', fontsize=14, fontweight='bold')
        ax.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_ylim(0, 1.1)
        plt.tight_layout()
        st.pyplot(fig)
    
    with tab2:
        st.subheader("Confusion Matrices for All 6 Models")
        
        # Display confusion matrices in 2 rows of 3
        col1, col2, col3 = st.columns(3)
        cols = [col1, col2, col3]
        
        model_names = list(all_predictions.keys())
        
        for i, name in enumerate(model_names[:3]):
            with cols[i]:
                fig = plot_confusion_matrix(y_test, all_predictions[name]['y_pred'], name)
                st.pyplot(fig)
        
        col4, col5, col6 = st.columns(3)
        cols2 = [col4, col5, col6]
        
        for i, name in enumerate(model_names[3:]):
            with cols2[i]:
                fig = plot_confusion_matrix(y_test, all_predictions[name]['y_pred'], name)
                st.pyplot(fig)
    
    with tab3:
        observations = get_model_observations(all_results)
        st.markdown(observations)

else:
    st.info("Please upload a test data CSV file using the sidebar to view model evaluation metrics.")
    st.markdown("""
    **Expected CSV Format:**
    - The file should contain Wine Quality dataset features
    - Must include 'quality' column (values 1-10)
    - Features: fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, 
      free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol
    
    **Sample file:** Use `sample_test_data.csv` provided in the project folder.
    """)
