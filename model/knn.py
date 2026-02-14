"""
K-Nearest Neighbor Classifier for Wine Quality Prediction
"""
from sklearn.neighbors import KNeighborsClassifier

def get_model():
    """Return a KNN model instance"""
    return KNeighborsClassifier(n_neighbors=5)

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
    'name': 'K-Nearest Neighbor',
    'type': 'Instance-based',
    'description': 'Classifies based on majority vote of k nearest neighbors. Sensitive to feature scaling and distance metric.',
    'hyperparameters': {
        'n_neighbors': 5
    }
}
