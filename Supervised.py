import numpy as np
import pandas as pd 
df = pd.read_csv('/kaggle/input/breast-cancer-dataset/Breast_cancer_dataset.csv')
columns_to_drop = ['id', 'diagnosis', 'Unnamed: 32']
X = df.drop(columns_to_drop, axis=1)
y = df['diagnosis']
y_num = pd.get_dummies(y, dtype=int)
y_num = y_num.drop('B', axis=1)
y_num = y_num.values.ravel()
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
scaler = StandardScaler()
X_test, X_train, y_test, y_train = train_test_split(X, y_num, test_size=0.3, stratify=y_num, random_state=1)
X_train_preprocessed = scaler.fit_transform(X_train)
X_test_preprocessed = scaler.fit_transform(X_test)
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, KFold
kf = KFold(n_splits=splits, shuffle=True, random_state=1)
knn_param_grid = {'n_neighbors': range(1,26)}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, knn_param_grid, cv=kf)
knn_cv.fit(X_train_preprocessed, y_train)
print(knn_cv.best_params_, knn_cv.best_score_)
import matplotlib.pyplot as plt
knn_results_df = knn_cv.cv_results_['mean_test_score']
n_neighbors = knn_param_grid['n_neighbors']
plt.figure(figsize=(6,5))
plt.plot(n_neighbors, knn_results_df, marker='o')
plt.xlabel('Number of neighbors')
plt.ylabel('Cross-validated accuracy')
plt.title('KNN Hyperparameter Tuning using GridSearchCV')
plt.xticks(n_neighbors)
plt.show()
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
knn_y_pred = knn_cv.predict(X_test_preprocessed)
conf_matrix = confusion_matrix(y_test, knn_y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['Benign', 'Malignant'])
disp.plot(cmap=plt.cm.Blues)
plt.title('KNeighbors Confusion Matrix')
plt.show()
print(classification_report(y_test, knn_y_pred))
from sklearn.linear_model import LogisticRegression
lr_param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'], 
                'max_iter' : [100, 200, 500, 1000]}
lr = LogisticRegression(solver='liblinear')
lr_cv = GridSearchCV(lr, lr_param_grid, cv=kf)
lr_cv.fit(X_train_preprocessed, y_train)
print(lr_cv.best_params_, lr_cv.best_score_)
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
lr_results_df = pd.DataFrame(lr_cv.cv_results_)
sns.lineplot(data=lr_results_df, x='param_C', y='mean_test_score', hue='param_penalty', marker='o')
plt.xscale('log')
plt.xlabel('Regularization Strength (value of C)')
plt.ylabel('Mean Test Score')
plt.grid(True)
plt.title('Logistic Regression Hyperparameter Tuning')
plt.show()
lr_y_pred = lr_cv.predict(X_test_preprocessed)
conf_matrix = confusion_matrix(y_test, lr_y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['Benign', 'Malignant'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Logistic Regression Confusion Matrix')
plt.show()
print(classification_report(y_test, lr_y_pred))
