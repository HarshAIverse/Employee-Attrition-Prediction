import os
import pandas as pd
import numpy as np
import joblib
from data_processing import DataProcessor

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline as SklearnPipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def build_preprocessor(num_cols, cat_cols):
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)
        ])
    return preprocessor

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
    
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_pred_proba),
        'Confusion Matrix': confusion_matrix(y_test, y_pred).tolist()
    }
    return metrics

def train_multiple_models(X_train, y_train, preprocessor):
    print("Training models...")
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    }
    
    trained_pipelines = {}
    
    for name, clf in models.items():
        # Using ImbPipeline to integrate SMOTE properly during cross-validation/training
        pipeline = ImbPipeline(steps=[
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', clf)
        ])
        
        # Fit model
        pipeline.fit(X_train, y_train)
        trained_pipelines[name] = pipeline
        print(f"[{name}] Training completed.")
        
    return trained_pipelines

def main():
    file_path = 'Palo Alto Networks.csv'
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return
    
    dp = DataProcessor(file_path)
    dp.load_data()
    dp.feature_engineering()
    
    X, y, num_cols, cat_cols = dp.get_features_and_target()
    
    print(f"Dataset shape: {X.shape}")
    print(f"Class distribution:\n{y.value_counts()}")
    
    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    preprocessor = build_preprocessor(num_cols, cat_cols)
    
    trained_models = train_multiple_models(X_train, y_train, preprocessor)
    
    best_model_name = None
    best_f1 = 0
    best_pipeline = None
    
    print("\nModel Evaluation on Test Set:")
    for name, pipeline in trained_models.items():
        metrics = evaluate_model(pipeline, X_test, y_test)
        print(f"\n--- {name} ---")
        for k, v in metrics.items():
            if k != 'Confusion Matrix':
                print(f"{k}: {v:.4f}")
        
        if metrics['F1 Score'] > best_f1:
            best_f1 = metrics['F1 Score']
            best_model_name = name
            best_pipeline = pipeline

    print(f"\nBest Model Selected: {best_model_name} (F1 Score: {best_f1:.4f})")
    
    # Optional: Hyperparameter tuning for XGBoost or Forest if selected
    # For demonstration we save the best base model but in production we'd gridsearch here.
    
    # Save the pipeline
    joblib.dump(best_pipeline, 'best_model.pkl')
    print("Saved 'best_model.pkl' to disk.")
    
    # Save the feature columns list for later use in explanation/UI
    feature_meta = {
        'num_cols': num_cols,
        'cat_cols': cat_cols,
        'all_cols': X.columns.tolist()
    }
    joblib.dump(feature_meta, 'feature_meta.pkl')
    
    # Save a clean dataset sample for explainability panel
    dp.data.to_csv('processed_data.csv', index=False)

if __name__ == "__main__":
    main()
