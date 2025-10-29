"""
Lesson 3-1：多模型比較（教學版）

本程式將使用 Wine 資料集，一次訓練 5 種模型：
1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Decision Tree
4. Random Forest
5. Support Vector Machine (SVM)

並比較各模型在測試集上的準確率（Accuracy）
"""

# ====== 1. 載入套件 ======
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# ====== 2. 載入資料 ======
data = load_wine()
X = data.data
y = data.target
print("資料筆數：", X.shape[0])
print("特徵數：", X.shape[1])
print("類別數：", len(set(y)))

# ====== 3. 切分訓練 & 測試資料 ======
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ====== 4. 特徵縮放（對 SVM、Logistic、KNN 有幫助） ======
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ====== 5. 建立模型清單 ======
models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM (RBF Kernel)": SVC(kernel='rbf')
}

# ====== 6. 訓練並評估每個模型 ======
print("\n===== 各模型 Accuracy 比較 =====")
results = {}  # 用來儲存結果

for name, model in models.items():
    # 注意：部分模型需用縮放後資料（Logistic/KNN/SVM）
    if name in ["Logistic Regression", "KNN (k=5)", "SVM (RBF Kernel)"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"{name:20s} → Accuracy: {acc:.4f}")

# ====== 7. 顯示最準確模型 ======
best_model = max(results, key=results.get)
print("\n✅ 最佳模型為：", best_model, f"（Accuracy = {results[best_model]:.4f}）")