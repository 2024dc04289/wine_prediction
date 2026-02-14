"""
Logistic Regression Model for Wine Quality Prediction
"""
from sklearn.linear_model import LogisticRegression

def get_model():
    """Return a Logistic Regression model instance"""
    return LogisticRegression(max_iter=1000, random_state=42)

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
    'name': 'Logistic Regression',
    'type': 'Linear',
    'description': 'Linear model for binary classification. Works well for linearly separable data. Provides interpretable coefficients.',
    'hyperparameters': {
        'max_iter': 1000,
        'random_state': 42
    }
}
