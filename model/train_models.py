"""
ML Assignment 2 - Classification Models Training
Dataset: Wine Quality Dataset (UCI Repository)
Problem: Predict wine quality (binary classification: quality >= 6 is High Quality)

Features: 11 physicochemical properties
Instances: 1599 samples

This script trains all 6 models using the individual model .py files.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                             recall_score, f1_score, matthews_corrcoef)

# Import model modules
import logistic_regression
import decision_tree
import knn
import naive_bayes
import random_forest
import xgboost_model

import warnings
warnings.filterwarnings('ignore')


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train all 6 models using individual .py modules and calculate evaluation metrics"""
    
    # Model modules mapping
    model_modules = {
        'Logistic Regression': logistic_regression,
        'Decision Tree': decision_tree,
        'K-Nearest Neighbor': knn,
        'Naive Bayes': naive_bayes,
        'Random Forest': random_forest,
        'XGBoost': xgboost_model
    }
    
    results = {}
    trained_models = {}
    
    for name, module in model_modules.items():
        print(f"\nTraining {name}...")
        
        # Train using module's train function
        model = module.train(X_train, y_train)
        trained_models[name] = model
        
        # Predict
        y_pred = module.predict(model, X_test)
        y_prob = module.predict_proba(model, X_test)
        
        # Calculate metrics
        metrics = {
            'Accuracy': round(accuracy_score(y_test, y_pred), 4),
            'AUC': round(roc_auc_score(y_test, y_prob), 4),
            'Precision': round(precision_score(y_test, y_pred, average='weighted'), 4),
            'Recall': round(recall_score(y_test, y_pred, average='weighted'), 4),
            'F1': round(f1_score(y_test, y_pred, average='weighted'), 4),
            'MCC': round(matthews_corrcoef(y_test, y_pred), 4)
        }
        
        results[name] = metrics
        print(f"  Accuracy: {metrics['Accuracy']}, AUC: {metrics['AUC']}, F1: {metrics['F1']}")
    
    return results, trained_models

def print_comparison_table(results):
    """Print comparison table of all models"""
    print("\n" + "="*90)
    print("MODEL COMPARISON TABLE")
    print("="*90)
    print(f"{'ML Model Name':<25} {'Accuracy':<10} {'AUC':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'MCC':<10}")
    print("-"*90)
    
    for model_name, metrics in results.items():
        print(f"{model_name:<25} {metrics['Accuracy']:<10} {metrics['AUC']:<10} {metrics['Precision']:<10} {metrics['Recall']:<10} {metrics['F1']:<10} {metrics['MCC']:<10}")
    
    print("="*90)

def get_observations(results):
    """Generate observations for each model"""
    observations = {}
    
    # Find best and worst performers
    best_accuracy = max(results.items(), key=lambda x: x[1]['Accuracy'])[0]
    best_f1 = max(results.items(), key=lambda x: x[1]['F1'])[0]
    best_auc = max(results.items(), key=lambda x: x[1]['AUC'])[0]
    
    observations['Logistic Regression'] = f"Provides a good baseline with {results['Logistic Regression']['Accuracy']} accuracy. Being a linear model, it performs well on linearly separable data and offers high interpretability. Fast training and prediction times."
    
    observations['Decision Tree'] = f"Achieves {results['Decision Tree']['Accuracy']} accuracy. Highly interpretable but prone to overfitting. The model captures non-linear patterns but may have high variance."
    
    observations['K-Nearest Neighbor'] = f"Achieves {results['K-Nearest Neighbor']['Accuracy']} accuracy. Performance depends on feature scaling and k value. Computationally expensive for large datasets during prediction."
    
    observations['Naive Bayes'] = f"Achieves {results['Naive Bayes']['Accuracy']} accuracy. Fast training and works well with high-dimensional data. Assumes feature independence which may limit performance on correlated features."
    
    observations['Random Forest'] = f"Achieves {results['Random Forest']['Accuracy']} accuracy with AUC of {results['Random Forest']['AUC']}. Ensemble approach reduces overfitting compared to single Decision Tree. Provides feature importance rankings."
    
    observations['XGBoost'] = f"Achieves {results['XGBoost']['Accuracy']} accuracy with AUC of {results['XGBoost']['AUC']}. Gradient boosting typically gives best performance. Handles missing values and provides regularization to prevent overfitting."
    
    return observations

if __name__ == "__main__":
    print("="*60)
    print("ML Assignment 2 - Classification Models Training")
    print("="*60)
    
    # Load data
    print("\nLoading dataset...")
    
    # Using Wine Quality dataset for demonstration
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    df = pd.read_csv(url, sep=';')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {df.columns.tolist()}")
    
    # Convert to binary classification (quality >= 6 is good wine)
    df['target'] = (df['quality'] >= 6).astype(int)
    
    # Features and target
    X = df.drop(['quality', 'target'], axis=1)
    y = df['target']
    
    print(f"\nNumber of features: {X.shape[1]}")
    print(f"Number of instances: {X.shape[0]}")
    print(f"Class distribution: {y.value_counts().to_dict()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models and get results (using individual .py model files)
    results, trained_models = train_and_evaluate_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Print comparison table
    print_comparison_table(results)
    
    # Get observations
    observations = get_observations(results)
    
    print("\n" + "="*90)
    print("MODEL OBSERVATIONS")
    print("="*90)
    for model_name, obs in observations.items():
        print(f"\n{model_name}:")
        print(f"  {obs}")
    
    # Save results
    results_df = pd.DataFrame(results).T
    results_df.to_csv('model_results.csv')
    
    print("\n\nAll models trained successfully!")
    print("Model files: logistic_regression.py, decision_tree.py, knn.py, naive_bayes.py, random_forest.py, xgboost_model.py")
    print("Results saved: model_results.csv")
