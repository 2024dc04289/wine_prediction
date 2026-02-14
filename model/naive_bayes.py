"""
Naive Bayes (Gaussian) Classifier for Wine Quality Prediction
"""
from sklearn.naive_bayes import GaussianNB

def get_model():
    """Return a Gaussian Naive Bayes model instance"""
    return GaussianNB()

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
    'name': 'Naive Bayes (Gaussian)',
    'type': 'Probabilistic',
    'description': 'Probabilistic classifier assuming feature independence and Gaussian distribution. Fast training, works well with limited data.',
    'hyperparameters': {}
}
