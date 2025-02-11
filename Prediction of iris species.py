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

model_dt = DecisionTreeClassifier(random_state=42)

model_dt.fit(X_train, y_train)

joblib.dump(model_dt, 'svm_model.joblib')




