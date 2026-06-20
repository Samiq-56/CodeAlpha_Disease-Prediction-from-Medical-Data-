# ============================================
# Disease Prediction — Breast Cancer
# CodeAlpha ML Internship — Task 4
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)
from sklearn.datasets import load_breast_cancer
import warnings
warnings.filterwarnings('ignore')

print("="*55)
print("   Breast Cancer Prediction")
print("="*55)

# ── 1. Load Dataset ───────────────────────────────────────
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target  # 0=malignant, 1=benign

print(f"Dataset shape: {df.shape}")
print(f"Benign: {df['target'].sum()} | Malignant: {(df['target']==0).sum()}")

# ── 2. EDA ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

pd.Series(data.target).value_counts().plot(kind='bar', ax=axes[0],
    color=['#e74c3c','#2ecc71'], edgecolor='black')
axes[0].set_title('Breast Cancer Distribution')
axes[0].set_xticklabels(['Malignant', 'Benign'], rotation=0)
axes[0].set_ylabel('Count')

top_features = df.corr(numeric_only=True)['target'].abs().sort_values(ascending=False)[1:11].index
sns.heatmap(df[top_features].corr(), annot=False, cmap='coolwarm', ax=axes[1])
axes[1].set_title('Top 10 Features Correlation')

plt.tight_layout()
plt.savefig('cancer_eda.png', dpi=100)
plt.show()
print("EDA saved!")

# ── 3. Preprocessing ──────────────────────────────────────
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ── 4. Train Models ───────────────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM':                 SVC(probability=True, random_state=42),
    'XGBoost':             XGBClassifier(eval_metric='logloss', random_state=42)
}

results = {}
print("\nModel Results:")
print("-"*50)

for name, model in models.items():
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cv  = cross_val_score(model, X_train_s, y_train, cv=5).mean()
    results[name] = {'accuracy': acc, 'auc': auc, 'cv': cv,
                     'model': model, 'y_pred': y_pred, 'y_prob': y_prob}
    print(f"{name:22} | Acc: {acc:.4f} | AUC: {auc:.4f} | CV: {cv:.4f}")

best_name = max(results, key=lambda x: results[x]['auc'])
best = results[best_name]
print(f"\nBest Model: {best_name} (AUC: {best['auc']:.4f})")

print(f"\nClassification Report — {best_name}:")
print(classification_report(y_test, best['y_pred'],
      target_names=['Malignant', 'Benign']))

# ── 5. Plots ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

names = list(results.keys())
accs  = [results[n]['accuracy'] for n in names]
axes[0].bar(names, accs, color=['#3498db','#2ecc71','#e74c3c','#f39c12'],
            edgecolor='black')
axes[0].set_title('Model Accuracy Comparison')
axes[0].set_ylabel('Accuracy')
axes[0].set_ylim(0.8, 1.0)
axes[0].tick_params(axis='x', rotation=15)

cm = confusion_matrix(y_test, best['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=axes[1],
            xticklabels=['Malignant','Benign'],
            yticklabels=['Malignant','Benign'])
axes[1].set_title(f'Confusion Matrix — {best_name}')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

for name, r in results.items():
    fpr, tpr, _ = roc_curve(y_test, r['y_prob'])
    axes[2].plot(fpr, tpr, label=f"{name} (AUC={r['auc']:.2f})")
axes[2].plot([0,1],[0,1],'k--')
axes[2].set_title('ROC Curves')
axes[2].set_xlabel('False Positive Rate')
axes[2].set_ylabel('True Positive Rate')
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig('cancer_results.png', dpi=100)
plt.show()

rf = results['Random Forest']['model']
feat_imp = pd.Series(rf.feature_importances_,
           index=data.feature_names).sort_values(ascending=False).head(15)
plt.figure(figsize=(12, 5))
feat_imp.plot(kind='bar', color='purple', edgecolor='black')
plt.title('Top 15 Feature Importance — Random Forest (Breast Cancer)')
plt.ylabel('Importance')
plt.tight_layout()
plt.savefig('cancer_feature_importance.png', dpi=100)
plt.show()
print("\nBreast Cancer Prediction — COMPLETE!")