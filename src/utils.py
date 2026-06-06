from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import os
import numpy as np

def evaluate_classifier(model, X, y):
    """
    Evaluates a supervised classifier.
    Returns:
        metrics: dict of classification report
        cm: confusion matrix
        roc_auc: ROC-AUC score
        y_pred: predicted labels
        y_prob: prediction probabilities for class 1
    """
    y_pred = model.predict(X)
    
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
        try:
            roc_auc = roc_auc_score(y, y_prob)
        except ValueError:
            roc_auc = 0.5
    else:
        y_prob = y_pred
        roc_auc = roc_auc_score(y, y_pred)
        
    report = classification_report(y, y_pred, output_dict=True)
    cm = confusion_matrix(y, y_pred)
    
    return {
        'report': report,
        'cm': cm,
        'roc_auc': roc_auc,
        'y_pred': y_pred,
        'y_prob': y_prob
    }

def evaluate_anomaly_detector(model, X, y):
    """
    Evaluates an anomaly detection model (Isolation Forest or One-Class SVM).
    Maps model outputs: 1 (inlier) -> 0 (normal), -1 (outlier) -> 1 (anomaly).
    Returns:
        metrics: dict of classification report
        cm: confusion matrix
        roc_auc: ROC-AUC score
        y_pred: predicted labels mapped to 0/1
        y_prob: anomaly scores used as probability proxy
    """
    # model.predict returns 1 for inlier, -1 for outlier
    raw_pred = model.predict(X)
    y_pred = np.where(raw_pred == -1, 1, 0)
    
    # decision_function: greater values represent inliers, lower values outliers.
    # So we take -decision_function as the anomaly score.
    scores = -model.decision_function(X)
    
    # Scale scores to [0, 1] range to act as pseudo-probabilities for display
    if scores.max() != scores.min():
        y_prob = (scores - scores.min()) / (scores.max() - scores.min())
    else:
        y_prob = scores
        
    try:
        roc_auc = roc_auc_score(y, y_prob)
    except ValueError:
        roc_auc = 0.5
        
    report = classification_report(y, y_pred, output_dict=True)
    cm = confusion_matrix(y, y_pred)
    
    return {
        'report': report,
        'cm': cm,
        'roc_auc': roc_auc,
        'y_pred': y_pred,
        'y_prob': y_prob
    }

def save_artifact(obj, file_path):
    """
    Saves an object (model, pipeline, etc.) using joblib.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump(obj, file_path)
    print(f"Saved artifact to {file_path}")

def load_artifact(file_path):
    """
    Loads an object using joblib.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Artifact not found: {file_path}")
    return joblib.load(file_path)
