from sklearn.datasets import load_wine
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
import seaborn as sns

# ---------- 共用：資料載入與前處理 ----------
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df["target"] = wine.target

X = df.iloc[:, :-1]
y = df["target"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

def evaluate_model(name, model):
    # 訓練
    model.fit(X_train, y_train)

    # 預測
    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)

    # 分數
    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test  = accuracy_score(y_test, y_pred_test)

    # 交叉驗證（快速看泛化穩定度）
    cv_scores = cross_val_score(model, X_scaled, y, cv=5)
    cv_mean, cv_std = cv_scores.mean(), cv_scores.std()

    # 報告
    print(f"\n======== {name} ========")
    print(f"Train Accuracy: {acc_train:.4f}")
    print(f"Test  Accuracy: {acc_test:.4f}")
    print(f"CV (5-fold)    : {cv_mean:.4f} ± {cv_std:.4f}")
    print("\nClassification Report (Test):")
    print(classification_report(y_test, y_pred_test, target_names=wine.target_names))

    # 混淆矩陣
    cm = confusion_matrix(y_test, y_pred_test)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    png_name = f"cm_{name.replace(' ','_').lower()}.png"
    plt.savefig(png_name)
    plt.close()
    print(f"📁 Confusion matrix saved: {png_name}")

    # 簡易過擬合判讀
    gap = acc_train - acc_test
    if gap > 0.05 and acc_train >= 0.98:
        print("⚠️ 可能過擬合：Train >> Test（差距明顯）")
    else:
        print("✅ 泛化看起來OK（Train 與 Test 接近）")

# ---------- 實驗 A：未限制的 Decision Tree（較容易過擬合） ----------
dt_unlimited = DecisionTreeClassifier(random_state=42)
evaluate_model("Decision Tree (unlimited)", dt_unlimited)

# ---------- 實驗 B：加上深度限制的 Decision Tree（抑制過擬合） ----------
dt_limited = DecisionTreeClassifier(max_depth=3, random_state=42)
evaluate_model("Decision Tree (max_depth=3)", dt_limited)

# ---------- 對照：Random Forest（通常較穩定） ----------
rf = RandomForestClassifier(random_state=42)
evaluate_model("Random Forest (baseline)", rf)

print("\n✅ 建議觀察重點：")
print("1) Decision Tree 未限制 vs max_depth=3：Train/Test差距縮小了嗎？")
print("2) Random Forest 與 Decision Tree 的 Test Accuracy / F1 誰較穩定？")
print("3) Confusion Matrix：哪個類別最容易被誤判？")