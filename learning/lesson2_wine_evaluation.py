"""
Lesson 2: Model Evaluation & Overfitting（模型評估與過擬合）

本課目標:
- 理解為什麼不能只看 Accuracy
- 學會使用 Confusion Matrix 與 Classification Report
- 學會判斷模型是否過擬合 (Overfitting)
- 練習比較不同模型表現

This lesson teaches:
- Why accuracy alone is not enough to evaluate a model
- How to use confusion matrix & classification report
- How to detect model overfitting
"""

from sklearn.datasets import load_wine
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load & Prepare Dataset
# -----------------------------
print("=== Step 1: Load Dataset 載入資料 ===")

wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df["target"] = wine.target

print(df.head())
print("\nTarget classes (類別分布):")
print(df["target"].value_counts())

# Split features & target
X = df.iloc[:, :-1]
y = df["target"]

# Standardize features (標準化處理，讓不同尺度的特徵一致)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -----------------------------
# 2. Train Model
# -----------------------------
print("\n=== Step 2: Train Model 訓練模型 ===")

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate Accuracy
test_acc = accuracy_score(y_test, y_pred)
print(f"✅ Test Accuracy 測試準確率: {test_acc:.4f}")

# -----------------------------
# 3. Confusion Matrix
# -----------------------------
print("\n=== Step 3: Confusion Matrix 混淆矩陣 ===")

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted 預測值")
plt.ylabel("Actual 真實值")
plt.tight_layout()
plt.savefig("wine_confusion_matrix.png")
plt.show()

print("📁 已輸出圖片: wine_confusion_matrix.png")

# -----------------------------
# 4. Classification Report
# -----------------------------
print("\n=== Step 4: Classification Report 分類評估報告 ===")

report = classification_report(y_test, y_pred, target_names=wine.target_names)
print(report)

print("""
📌 指標解讀:
Precision：預測為該類別的樣本中，有多少是真的（準不準）
Recall：實際該類別的樣本，有多少被模型找出來（有沒有漏）
F1-score：Precision 與 Recall 的平衡分數（越高越好）
""")

# -----------------------------
# 5. Check Overfitting 檢查是否過擬合
# -----------------------------
print("\n=== Step 5: Check Overfitting 過擬合檢測 ===")

train_pred = model.predict(X_train)
train_acc = accuracy_score(y_train, train_pred)

print(f"Train Accuracy 訓練準確率: {train_acc:.4f}")
print(f"Test Accuracy 測試準確率: {test_acc:.4f}")

if train_acc > test_acc + 0.05:
    print("⚠️ 模型有過擬合傾向 Model is likely overfitting.")
else:
    print("✅ 模型表現健康 Model is well-generalized.")

# -----------------------------
# 6. Lesson Practice 練習題
# -----------------------------
print("""
=== ✅ 小練習 Practice ===
請嘗試做以下練習 (Try this):

1. 將模型改成 DecisionTreeClassifier，重新比較以下項目:
   - Accuracy 是否變化？
   - 混淆矩陣有何差異？
   - Train vs Test Accuracy 是否更容易過擬合？

提示:
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=42)

完成後，你會比現在更理解「為何 RandomForest 通常比 Decision Tree 好」。
""")
