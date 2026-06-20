# 🏥 Disease Prediction from Medical Data — CodeAlpha ML Internship

ML models that predict diseases using patient data. Three datasets covered with 4 algorithms each.

---

## 📌 Task Info
- **Internship:** CodeAlpha Machine Learning
- **Task:** Task 4 — Disease Prediction from Medical Data
- **Algorithms:** Logistic Regression, Random Forest, SVM, XGBoost

---

## 📊 Results Summary

| Disease | Dataset Size | Best Model | Accuracy | AUC |
|---------|-------------|-----------|----------|-----|
| Heart Disease | 270 patients | Logistic Regression | **85.19%** | 0.8986 |
| Diabetes | 768 patients | Logistic Regression | **71.43%** | 0.8230 |
| Breast Cancer | 569 patients | Logistic Regression | **98.25%** | 0.9954 |

---

## 🚀 How to Run

```bash
pip install pandas scikit-learn xgboost matplotlib seaborn

python heart_disease.py
python diabetes.py
python breast_cancer.py
```

> Datasets load automatically — no manual download needed.

---

## 📁 Files

| File | Disease | Dataset |
|------|---------|---------|
| `heart_disease.py` | Heart Disease | UCI Heart Statlog |
| `diabetes.py` | Diabetes | Pima Indians Diabetes |
| `breast_cancer.py` | Breast Cancer | Wisconsin Breast Cancer |

---

## 📈 Outputs Generated

Each script produces:
- EDA plots (distribution + correlation)
- Model accuracy comparison chart
- Confusion matrix
- ROC curves
- Feature importance chart

---

## 🛠 Tech Stack
Python • Scikit-learn • XGBoost • Pandas • Matplotlib • Seaborn

---

*CodeAlpha ML Internship — Task 4*
