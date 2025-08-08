import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings('ignore')

df = pd.read_csv("/kaggle/input/breast-cancer-dataset/Breast_cancer_dataset.csv")

df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

plt.figure(figsize=(6,4))
sns.countplot(x='diagnosis', data=df, palette='coolwarm')
plt.title('Diagnosis Count (0 = Benign, 1 = Malignant)')
plt.xlabel('Diagnosis')
plt.ylabel('Count')
plt.xticks([0, 1], ['Benign', 'Malignant'])
plt.show()

plt.figure(figsize=(20, 20))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()

features_to_plot = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'concavity_mean']

plt.figure(figsize=(15, 10))
for i, feature in enumerate(features_to_plot):
    plt.subplot(2, 3, i+1)
    sns.histplot(data=df, x=feature, hue='diagnosis', kde=True, palette='coolwarm')
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 10))
for i, feature in enumerate(features_to_plot):
    plt.subplot(2, 3, i+1)
    sns.boxplot(x='diagnosis', y=feature, data=df, palette='Set2')
    plt.title(f'{feature} by Diagnosis')
    plt.xticks([0, 1], ['Benign', 'Malignant'])
plt.tight_layout()
plt.show()

selected_features = ['radius_mean', 'texture_mean', 'area_mean', 'smoothness_mean', 'diagnosis']
sns.pairplot(df[selected_features], hue='diagnosis', palette='husl')
plt.suptitle('Pairplot of Selected Features', y=1.02)
plt.show()

df = df.drop(columns=['id', 'Unnamed: 32'])
df['diagnosis'] = LabelEncoder().fit_transform(df['diagnosis'])

X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC(),
    "K-Nearest Neighbors": KNeighborsClassifier()
}

accuracies = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    accuracies[name] = round(accuracy * 100, 2)

print("Model Accuracies (%):")
for name, acc in accuracies.items():
    print(f"{name}: {acc}%")

plt.figure(figsize=(10, 5))
plt.bar(accuracies.keys(), accuracies.values(), color='skyblue')
plt.ylabel('Accuracy (%)')
plt.title('Comparison of ML Models on Breast Cancer Dataset')
plt.xticks(rotation=45)
plt.ylim(80, 100)
plt.grid(axis='y')
plt.tight_layout()
plt.show()
