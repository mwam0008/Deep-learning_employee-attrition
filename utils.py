"""
utils.py - Visualization helpers
"""
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def plot_attrition_distribution(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df['Attrition'].value_counts()
    ax.bar(counts.index.astype(str), counts.values, color=['#2196F3', '#F44336'])
    ax.set_title('Employee Attrition Distribution')
    ax.set_xlabel('Attrition (0 = Stayed, 1 = Left)')
    ax.set_ylabel('Count')
    for i, v in enumerate(counts.values):
        ax.text(i, v + 5, str(v), ha='center', fontweight='bold')
    plt.tight_layout()
    return fig

def plot_training_curves(loss_curve, acc_curve):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(loss_curve, color='#F44336', label='Loss')
    axes[0].set_title('Training Loss Over Epochs')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[1].plot(acc_curve, color='#2196F3', label='Accuracy')
    axes[1].set_title('Training Accuracy Over Epochs')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    plt.tight_layout()
    return fig

def plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Stayed (0)', 'Left (1)'],
                yticklabels=['Stayed (0)', 'Left (1)'], ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    plt.tight_layout()
    return fig

def plot_learning_rate_search(lrs, losses):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogx(lrs, losses, color='#9C27B0', marker='o')
    ax.set_xlabel('Learning Rate (log scale)')
    ax.set_ylabel('Loss')
    ax.set_title('Learning Rate vs Loss — Find the Sweet Spot')
    ax.axvline(x=0.01, color='red', linestyle='--', label='~Good LR zone')
    ax.legend()
    plt.tight_layout()
    return fig

def plot_feature_correlation(df):
    numeric_df = df.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(numeric_df.corr(), cmap='coolwarm', ax=ax, linewidths=0.5)
    ax.set_title('Feature Correlation Heatmap')
    plt.tight_layout()
    return fig
