"""
XGBoost Classifier for Wine Quality Prediction (Ensemble Method)
"""
from xgboost import XGBClassifier

def get_model():
    """Return an XGBoost model instance"""
    return XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')

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
    'name': 'XGBoost',
    'type': 'Ensemble (Gradient Boosting)',
    'description': 'Gradient boosting algorithm with regularization. Excellent performance on structured/tabular data. Handles missing values.',
    'hyperparameters': {
        'n_estimators': 100,
        'random_state': 42,
        'eval_metric': 'logloss'
    }
}
