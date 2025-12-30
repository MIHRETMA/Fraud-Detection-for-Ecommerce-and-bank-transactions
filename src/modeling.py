import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (f1_score, confusion_matrix, average_precision_score)
from sklearn.ensemble import RandomForestClassifier


def feature_engineering(df, target_col):
    drop_cols = ['signup_time', 'purchase_time', 'device_id', 'user_id', 'ip_address']
    # Safely drop optional columns if present
    df = df.drop(columns=drop_cols, errors='ignore')

    # Ensure target exists
    if target_col not in df.columns:
        raise KeyError(f"target_col '{target_col}' not found in DataFrame columns: {list(df.columns)}")

    # Prepare features and label
    X = df.drop(columns=[target_col])
    y = df[target_col]

    if 'country' in df.columns:
        X = pd.get_dummies(X, columns=['country'], drop_first=True)

    return X, y

def split_data(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    return X_train, X_test, y_train, y_test

def baseline_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000 ,class_weight='balanced', n_jobs=1)
    model.fit(X_train, y_train)
    return model

def ensemble_model(model, X_train, y_train):
    if model == RandomForestClassifier:
        model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                class_weight='balanced',
                random_state=42,
                n_jobs=1
                )
    model.fit(X_train, y_train)
    return model

def model_evaluator(model, X_test, y_test):
    model_name = model.__class__.__name__
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    f1 = f1_score(y_test, y_pred)
    auc_pr = average_precision_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    
    print("Model Name: ", model_name)
    print("F1-score: ", f1)
    print("AUC-PR: ", auc_pr)
    print("Confusion Matric: \n", cm)
