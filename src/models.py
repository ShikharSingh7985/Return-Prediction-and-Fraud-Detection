from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import OneClassSVM
import numpy as np

def train_logistic_regression(X, y, random_state=42):
    """
    Trains a Logistic Regression classifier on X and y.
    """
    model = LogisticRegression(max_iter=1000, random_state=random_state, class_weight=None)
    model.fit(X, y)
    return model

def train_random_forest(X, y, random_state=42):
    """
    Trains a Random Forest classifier on X and y.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    model.fit(X, y)
    return model

def train_isolation_forest(X, contamination=0.05, random_state=42):
    """
    Trains an Isolation Forest anomaly detector on X.
    Contamination is the expected proportion of anomalies in the dataset.
    """
    model = IsolationForest(contamination=contamination, random_state=random_state, n_jobs=-1)
    model.fit(X)
    return model

def train_one_class_svm(X_normal, nu=0.05, kernel='rbf', gamma='scale', max_train_samples=5000, random_state=42):
    """
    Trains a One-Class SVM on normal data only (y == 0).
    Limits training size to max_train_samples to ensure fast execution.
    """
    # Downsample normal data if it exceeds max_train_samples to keep it lightweight
    if len(X_normal) > max_train_samples:
        rng = np.random.default_rng(random_state)
        indices = rng.choice(len(X_normal), max_train_samples, replace=False)
        X_train_subset = X_normal[indices]
    else:
        X_train_subset = X_normal
        
    model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma, cache_size=1000)
    model.fit(X_train_subset)
    return model
