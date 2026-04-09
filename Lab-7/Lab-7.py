import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error

# Classification Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# Regression
from sklearn.linear_model import LinearRegression

# Clustering
from sklearn.cluster import KMeans, DBSCAN
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt


df = pd.read_csv("C:/Games/Desktop/Documents/Projects_Sem4/ML/dataset/video_features.csv")

df = df.dropna()

X = df.drop("label", axis=1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

params = {
    "n_neighbors": list(range(1, 20)),
    "weights": ["uniform", "distance"]
}

search = RandomizedSearchCV(
    KNeighborsClassifier(),
    params,
    n_iter=5,
    cv=3,
    random_state=42
)

search.fit(X_train, y_train)
best_knn = search.best_estimator_

models = {
    "SVM": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Naive Bayes": GaussianNB(),
    "AdaBoost": AdaBoostClassifier(),
    "MLP": MLPClassifier(max_iter=500),
    "KNN": best_knn
}

results_list = []

for name, model in models.items():
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, zero_division=0)
    recall = recall_score(y_test, y_test_pred, zero_division=0)
    f1 = f1_score(y_test, y_test_pred, zero_division=0)

    results_list.append([
        name,
        train_acc,
        test_acc,
        precision,
        recall,
        f1
    ])

results_df = pd.DataFrame(results_list, columns=[
    "Model", "Train Accuracy", "Test Accuracy", "Precision", "Recall", "F1 Score"
])

print("\n=== CLASSIFICATION RESULTS ===")
print(results_df)

model_r = LinearRegression()
model_r.fit(X_train, y_train)

y_pred_r = model_r.predict(X_test)

mse = mean_squared_error(y_test, y_pred_r)

print("\n=== REGRESSION ===")
print("MSE:", mse)


kmeans = KMeans(n_clusters=2, random_state=42)
kmeans_labels = kmeans.fit_predict(X)

dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X)

Z = linkage(X, method='ward')

plt.figure(figsize=(6,4))
dendrogram(Z)
plt.title("Hierarchical Clustering")
plt.show()