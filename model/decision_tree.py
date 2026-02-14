"""
Decision Tree Classifier for Wine Quality Prediction
"""
from sklearn.tree import DecisionTreeClassifier

def get_model():
    """Return a Decision Tree model instance"""
    return DecisionTreeClassifier(random_state=42)

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
    'name': 'Decision Tree',
    'type': 'Non-linear',
    'description': 'Tree-based model that splits data based on feature thresholds. Highly interpretable but prone to overfitting.',
    'hyperparameters': {
        'random_state': 42
    }
}
