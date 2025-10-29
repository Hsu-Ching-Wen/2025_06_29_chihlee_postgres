from sklearn.datasets import load_wine
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. 載入資料
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['target'] = wine.target

print("\n📌 資料前 5 列：")
print(df.head())

print("\n📌 各類別筆數：")
print(df['target'].value_counts())

# 2. 類別分布圖
sns.countplot(data=df, x='target')
plt.savefig("wine_countplot.png")
plt.close()
print("✅ 類別分布圖：wine_countplot.png")

# 3. 特徵相關係數熱圖
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=False, cmap="coolwarm")
plt.savefig("wine_corr_heatmap.png")
plt.close()
print("✅ 特徵關聯熱圖：wine_corr_heatmap.png")

# 4. 機器學習模型訓練
X = df.iloc[:, :-1]
y = df['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("\n🎯 模型準確率：", acc)