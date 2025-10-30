"""
Lesson 3-2：多模型視覺化比較（完整）
輸出：
- accuracy_comparison.png
- prf_comparison.png   （Precision/Recall/F1 宏平均）
- f1_radar.png         （Top-3 模型在各類別的 F1 雷達圖）
- cm_[model].png       （各模型混淆矩陣）
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

# --------- 1) 資料與前處理 ---------
data = load_wine()
X, y, class_names = data.data, data.target, data.target_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# --------- 2) 模型定義 ---------
models = OrderedDict({
    "Logistic": LogisticRegression(max_iter=500),
    "KNN(k=5)": KNeighborsClassifier(n_neighbors=5),
    "KNN (k=7)": KNeighborsClassifier(n_neighbors=7),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(random_state=42),
    # ✅ 新增 Gradient Boosting（提升樹模型）
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    # ✅ 新增 XGBoost（常用穩健參數）
    "XGBoost": XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    n_jobs=-1,
    eval_metric="mlogloss"
    ),
    "SVM(RBF)": SVC(kernel="rbf", probability=False),
    # ✅ 新增 Linear SVM（線性核）
    "SVM (Linear)": SVC(kernel='linear'),
    #"Naive Bayes": GaussianNB()
})

def fit_predict(name, model):
    """依模型需求選擇是否使用縮放資料"""
    if name in ["Logistic", "KNN(k=5)", "SVM(RBF)"]:
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    return y_pred, model

# --------- 3) 訓練、評估、蒐集指標 ---------
acc_dict = {}
prf_dict = {}   # 宏平均 precision / recall / f1
f1_per_class = {}  # 各類別 f1（給雷達圖用）

os.makedirs("images", exist_ok=True)

def plot_cm(cm, labels, title, fname):
    plt.figure(figsize=(5.2, 4.2))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(fname, dpi=160)
    plt.show()

for name, model in models.items():
    y_pred, fitted = fit_predict(name, model)

    # 1) Accuracy
    acc = accuracy_score(y_test, y_pred)
    acc_dict[name] = acc

    # 2) Classification report → 宏平均與各類別 f1
    report = classification_report(
        y_test, y_pred, target_names=class_names, output_dict=True
    )
    prf_dict[name] = {
        "precision": report["macro avg"]["precision"],
        "recall":    report["macro avg"]["recall"],
        "f1":        report["macro avg"]["f1-score"]
    }
    f1_per_class[name] = [report[c]["f1-score"] for c in class_names]

    # 3) Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plot_cm(cm, class_names, f"Confusion Matrix – {name}",
            f"images/cm_{name.replace('(','').replace(')','').replace(' ','_').lower()}.png")

# --------- 4) 圖一：Accuracy 比較 ---------
plt.figure(figsize=(7.2, 4.2))
order = sorted(acc_dict, key=acc_dict.get, reverse=True)
vals = [acc_dict[k] for k in order]
sns.barplot(x=order, y=vals)
for i, v in enumerate(vals):
    plt.text(i, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
plt.ylim(0, 1.05)
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.tight_layout()
plt.savefig("images/accuracy_comparison.png", dpi=160)
plt.show()

print("\n[Accuracy]")
for k in order:
    print(f"{k:15s} : {acc_dict[k]:.4f}")

# --------- 5) 圖二：Precision/Recall/F1（宏平均） ---------
metrics = ["precision", "recall", "f1"]
plt.figure(figsize=(8.8, 4.6))
x = np.arange(len(models))
width = 0.26

for i, m in enumerate(metrics):
    plt.bar(x + (i-1)*width,
            [prf_dict[k][m] for k in models.keys()],
            width=width, label=m.capitalize())

plt.xticks(x, list(models.keys()))
plt.ylim(0, 1.05)
plt.ylabel("Macro-Avg Score")
plt.title("Macro Precision / Recall / F1 by Model")
plt.legend()
plt.tight_layout()
plt.savefig("images/prf_comparison.png", dpi=160)
plt.show()

print("\n[Macro Precision/Recall/F1]")
for k in models.keys():
    p, r, f = prf_dict[k]["precision"], prf_dict[k]["recall"], prf_dict[k]["f1"]
    print(f"{k:15s} : P={p:.4f}  R={r:.4f}  F1={f:.4f}")

# --------- 6) 圖三：F1 雷達圖（Top-3 模型，按 Macro-F1 排名） ---------
# 避免過度擁擠，僅選 Macro-F1 前三名
top3 = sorted(models.keys(), key=lambda k: prf_dict[k]["f1"], reverse=True)[:3]
labels = list(class_names)
angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
angles = np.concatenate([angles, [angles[0]]])  # 關閉雷達圈

plt.figure(figsize=(6.6, 6.6))
ax = plt.subplot(111, polar=True)

for name in top3:
    vals = f1_per_class[name]
    vals = np.concatenate([vals, [vals[0]]])  # 關閉雷達圈
    ax.plot(angles, vals, linewidth=2, label=name)
    ax.fill(angles, vals, alpha=0.15)

ax.set_thetagrids(angles[:-1] * 180/np.pi, labels)
ax.set_ylim(0, 1.05)
plt.title("Per-Class F1 Radar (Top-3 Models)")
plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1.12))
plt.tight_layout()
plt.savefig("images/f1_radar.png", dpi=160)
plt.show()

print("\n✅ 圖檔已輸出到 images/ 資料夾：")
print("- images/accuracy_comparison.png")
print("- images/prf_comparison.png")
print("- images/f1_radar.png")
print("- images/cm_*.png  （各模型的混淆矩陣）")