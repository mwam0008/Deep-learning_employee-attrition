"""
model.py - Neural Network for Employee Attrition Classification
Uses sklearn's MLPClassifier (same concept as TensorFlow, no install issues)
"""

import logging
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_and_prepare_data(filepath: str):
    """Load CSV and separate features from target (Attrition)."""
    try:
        logging.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        logging.info(f"Data loaded. Shape: {df.shape}")
        Y = df['Attrition']
        X = df.drop(columns=['Attrition'])
        return df, X, Y
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        raise


def scale_and_split(X, Y, test_size=0.2, random_state=1):
    """Scale features and split into train/test sets."""
    try:
        sc = StandardScaler()
        X_scaled = sc.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        x_train, x_test, y_train, y_test = train_test_split(
            X_scaled, Y, test_size=test_size, random_state=random_state, stratify=Y
        )
        logging.info(f"Train: {x_train.shape}, Test: {x_test.shape}")
        return x_train, x_test, y_train, y_test, sc
    except Exception as e:
        logging.error(f"Scaling/splitting failed: {e}")
        raise


def build_and_train_model(x_train, y_train, neurons=3, extra_layer=False,
                           learning_rate=0.01, epochs=100, activation='relu'):
    """Build and train a neural network using sklearn MLPClassifier."""
    try:
        hidden_layers = (neurons, neurons) if extra_layer else (neurons,)
        sk_activation = 'logistic' if activation == 'sigmoid' else activation

        logging.info(f"Training MLP: layers={hidden_layers}, lr={learning_rate}, epochs={epochs}")

        model = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation=sk_activation,
            learning_rate_init=learning_rate,
            max_iter=1,
            random_state=42,
            warm_start=True
        )

        loss_curve = []
        acc_curve = []
        for epoch in range(epochs):
            model.fit(x_train, y_train)
            loss_curve.append(model.loss_)
            acc_curve.append(accuracy_score(y_train, model.predict(x_train)))

        logging.info("Training complete.")
        return model, loss_curve, acc_curve
    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise


def evaluate_model(model, x_test, y_test):
    """Evaluate model on test set."""
    try:
        y_preds = model.predict(x_test)
        y_probs = model.predict_proba(x_test)[:, 1]
        acc = accuracy_score(y_test, y_preds)
        report = classification_report(y_test, y_preds, output_dict=True)
        cm = confusion_matrix(y_test, y_preds)
        logging.info(f"Test Accuracy: {acc:.4f}")
        return acc, report, cm, y_preds, y_probs
    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        raise


def find_best_learning_rate(x_train, y_train, neurons=3):
    """Test multiple learning rates and return loss for each."""
    try:
        lrs = np.logspace(-5, 0, 20)
        losses = []
        for lr in lrs:
            m = MLPClassifier(hidden_layer_sizes=(neurons,), learning_rate_init=lr,
                              max_iter=50, random_state=42)
            m.fit(x_train, y_train)
            losses.append(m.loss_)
        return lrs, losses
    except Exception as e:
        logging.error(f"LR search failed: {e}")
        raise
