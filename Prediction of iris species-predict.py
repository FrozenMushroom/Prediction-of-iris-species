from sklearn.datasets import load_iris
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

loaded_model = joblib.load('svm_model.joblib')

data = load_iris()

df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
df['species'] = df['target'].apply(lambda x: data.target_names[x])

X = df.drop(columns=['target', 'species'])
y = df['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

selector = SelectKBest(f_classif, k=2)
X_new = selector.fit_transform(X_scaled, y)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

y_pred = loaded_model.predict(X_test)

accuracy_dt = accuracy_score(y_test, y_pred)
print(f"Decision Tree Accuracy: {accuracy_dt:.4f}")