from sklearn.datasets import load_iris
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. 載入資料
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target

print("資料前五列：")
print(df.head())

# 2. 可視化（儲存為圖片）
sns.pairplot(df, hue='species')
plt.savefig("iris_pairplot.png")
plt.close()
print

# 3. 分割訓練/測試資料
X = df.iloc[:, :-1]
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. 訓練模型
model = KNeighborsClassifier()
model.fit(X_train, y_train)

# 5. 預測與評估
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("模型準確率：", acc)