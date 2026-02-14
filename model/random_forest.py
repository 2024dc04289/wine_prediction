"""
Random Forest Classifier for Wine Quality Prediction (Ensemble Method)
"""
from sklearn.ensemble import RandomForestClassifier

def get_model():
    """Return a Random Forest model instance"""
    return RandomForestClassifier(n_estimators=100, random_state=42)

def train(X_train, y_train):
    """Train the model and return fitted model"""
    model = get_model()
    model.fit(X_train, y_train)
    return model

def predict(model, X_test):
    """Make predictions"""
    return model.predict(X_test)

def predict_proba(model, X_test):
    """Get prediction probabilities"""
    return model.predict_proba(X_test)[:, 1]

# Model characteristics
MODEL_INFO = {
    'name': 'Random Forest',
    'type': 'Ensemble (Bagging)',
    'description': 'Ensemble of decision trees using bagging. Reduces overfitting and provides feature importance. Robust for tabular data.',
    'hyperparameters': {
        'n_estimators': 100,
        'random_state': 42
    }
}
