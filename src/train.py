import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Import project modules
from data_loader import load_and_preprocess_raw
from preprocessing import build_preprocessing_pipeline, apply_smote
from models import (
    train_logistic_regression,
    train_random_forest,
    train_isolation_forest,
    train_one_class_svm
)
from utils import evaluate_classifier, evaluate_anomaly_detector, save_artifact

def run_training_pipeline():
    # Define paths
    base_dir = "c:/Users/s4shi/Desktop/A fraud detection"
    trans_path = os.path.join(base_dir, "data", "train_transaction.csv")
    ident_path = os.path.join(base_dir, "data", "train_identity.csv")
    models_dir = os.path.join(base_dir, "models")
    
    # 1. Load and sample data
    print("--- Phase 1: Ingesting and sampling data ---")
    df, num_cols, cat_cols = load_and_preprocess_raw(trans_path, ident_path, sample_fraction=0.04, random_state=42)
    
    # 2. Split into Train & Test
    print("\n--- Phase 2: Splitting Train/Test sets ---")
    X = df.drop(columns=['isFraud', 'isReturned'])
    y_fraud = df['isFraud']
    y_returned = df['isReturned']
    
    # Split for Fraud
    X_train, X_test, y_train_fraud, y_test_fraud = train_test_split(
        X, y_fraud, test_size=0.2, random_state=42, stratify=y_fraud
    )
    
    # Split for Return
    _, _, y_train_returned, y_test_returned = train_test_split(
        X, y_returned, test_size=0.2, random_state=42, stratify=y_returned
    )
    
    # 3. Fit preprocessing pipeline
    print("\n--- Phase 3: Building preprocessing pipeline ---")
    preprocessor = build_preprocessing_pipeline(num_cols, cat_cols)
    preprocessor.fit(X_train)
    
    # Transform datasets
    X_train_prep = preprocessor.transform(X_train)
    X_test_prep = preprocessor.transform(X_test)
    
    print(f"Preprocessed train feature matrix shape: {X_train_prep.shape}")
    print(f"Preprocessed test feature matrix shape: {X_test_prep.shape}")
    
    # Save the preprocessor
    save_artifact(preprocessor, os.path.join(models_dir, "preprocessor.joblib"))
    
    # Save the feature names (useful for importance in app)
    # Get categorical feature names after one-hot encoding
    try:
        cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
        cat_features_encoded = cat_encoder.get_feature_names_out(cat_cols).tolist()
    except Exception:
        cat_features_encoded = []
    feature_names = num_cols + cat_features_encoded
    save_artifact(feature_names, os.path.join(models_dir, "feature_names.joblib"))
    
    # We will store performance metrics for all models here
    results = {
        'fraud': {},
        'returned': {}
    }
    
    targets = {
        'fraud': (y_train_fraud, y_test_fraud),
        'returned': (y_train_returned, y_test_returned)
    }
    
    # 4. Train and evaluate models for each target
    for target_name, (y_train, y_test) in targets.items():
        print(f"\n==========================================")
        print(f"Training models for target: {target_name.upper()}")
        print(f"==========================================")
        
        # Calculate positive class ratio in training set
        pos_ratio = y_train.mean()
        print(f"Positive class ratio (anomaly rate) in training: {pos_ratio*100:.2f}%")
        
        # A. Classifier WITHOUT SMOTE
        print("\n--- Training Supervised Classifiers (No SMOTE) ---")
        print("1. Training Logistic Regression...")
        lr_no_smote = train_logistic_regression(X_train_prep, y_train)
        lr_no_smote_eval = evaluate_classifier(lr_no_smote, X_test_prep, y_test)
        print(f"   ROC-AUC: {lr_no_smote_eval['roc_auc']:.4f} | Recall: {lr_no_smote_eval['report']['1']['recall']:.4f}")
        
        print("2. Training Random Forest...")
        rf_no_smote = train_random_forest(X_train_prep, y_train)
        rf_no_smote_eval = evaluate_classifier(rf_no_smote, X_test_prep, y_test)
        print(f"   ROC-AUC: {rf_no_smote_eval['roc_auc']:.4f} | Recall: {rf_no_smote_eval['report']['1']['recall']:.4f}")
        
        # B. Classifier WITH SMOTE
        print("\n--- Training Supervised Classifiers (WITH SMOTE) ---")
        print("Applying SMOTE to training data...")
        X_train_res, y_train_res = apply_smote(X_train_prep, y_train)
        print(f"   Balanced training shape: {X_train_res.shape} (Positives: {y_train_res.sum()})")
        
        print("1. Training Logistic Regression (SMOTE)...")
        lr_smote = train_logistic_regression(X_train_res, y_train_res)
        lr_smote_eval = evaluate_classifier(lr_smote, X_test_prep, y_test)
        print(f"   ROC-AUC: {lr_smote_eval['roc_auc']:.4f} | Recall: {lr_smote_eval['report']['1']['recall']:.4f}")
        
        print("2. Training Random Forest (SMOTE)...")
        rf_smote = train_random_forest(X_train_res, y_train_res)
        rf_smote_eval = evaluate_classifier(rf_smote, X_test_prep, y_test)
        print(f"   ROC-AUC: {rf_smote_eval['roc_auc']:.4f} | Recall: {rf_smote_eval['report']['1']['recall']:.4f}")
        
        # C. Anomaly Detection: Isolation Forest
        print("\n--- Training Isolation Forest ---")
        # Contamination matches the anomaly rate
        if_model = train_isolation_forest(X_train_prep, contamination=max(0.01, pos_ratio))
        if_eval = evaluate_anomaly_detector(if_model, X_test_prep, y_test)
        print(f"   ROC-AUC: {if_eval['roc_auc']:.4f} | Recall: {if_eval['report']['1']['recall']:.4f}")
        
        # D. Anomaly Detection: One-Class SVM
        print("\n--- Training One-Class SVM (Normal data only) ---")
        # OC-SVM is trained on NORMAL instances only (label 0)
        X_train_normal = X_train_prep[y_train == 0]
        # nu value represents the training error rate (contamination proxy)
        ocsvm_model = train_one_class_svm(X_train_normal, nu=max(0.01, pos_ratio))
        ocsvm_eval = evaluate_anomaly_detector(ocsvm_model, X_test_prep, y_test)
        print(f"   ROC-AUC: {ocsvm_eval['roc_auc']:.4f} | Recall: {ocsvm_eval['report']['1']['recall']:.4f}")
        
        # Save models
        save_artifact(lr_no_smote, os.path.join(models_dir, f"{target_name}_lr_no_smote.joblib"))
        save_artifact(rf_no_smote, os.path.join(models_dir, f"{target_name}_rf_no_smote.joblib"))
        save_artifact(lr_smote, os.path.join(models_dir, f"{target_name}_lr_smote.joblib"))
        save_artifact(rf_smote, os.path.join(models_dir, f"{target_name}_rf_smote.joblib"))
        save_artifact(if_model, os.path.join(models_dir, f"{target_name}_isolation_forest.joblib"))
        save_artifact(ocsvm_model, os.path.join(models_dir, f"{target_name}_one_class_svm.joblib"))
        
        # Record results for plotting
        results[target_name] = {
            'lr_no_smote': lr_no_smote_eval,
            'rf_no_smote': rf_no_smote_eval,
            'lr_smote': lr_smote_eval,
            'rf_smote': rf_smote_eval,
            'isolation_forest': if_eval,
            'one_class_svm': ocsvm_eval
        }
        
    # Save test set and target arrays for verification and graphing in dashboard
    # Since we want to make it fast, we can save a subset of test data and results
    # To avoid storing large arrays, we only save predictions/probabilities and metrics
    eval_package = {
        'results': results,
        'y_test_fraud': y_test_fraud.values,
        'y_test_returned': y_test_returned.values,
        'features_used': {
            'num': num_cols,
            'cat': cat_cols
        }
    }
    save_artifact(eval_package, os.path.join(models_dir, "evaluation_results.joblib"))
    print("\n--- Training Pipeline Completed Successfully! ---")

if __name__ == '__main__':
    run_training_pipeline()
