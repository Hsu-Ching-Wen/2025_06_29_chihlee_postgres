from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. 載入資料
data = load_wine()
X, y = data.data, data.target

# 2. 切分資料
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. KNN 需要標準化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. 測試多個 k
k_values = [1, 3, 5, 7, 9, 11]

print("🔍 KNN 調參：比較不同 k 值\n")
for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # 自動評語
    if acc >= 0.97:
        comment = "✅ 很好"
    elif acc >= 0.90:
        comment = "🟨 可接受"
    else:
        comment = "❌ 不佳"

    print(f"k={k} → Accuracy: {acc:.4f} {comment}")