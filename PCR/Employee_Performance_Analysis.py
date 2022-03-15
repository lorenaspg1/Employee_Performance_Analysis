# Data cleaning and preparation
# ==============================================================================
import multiprocessing

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

pd.set_option('display.max_columns', None)

# Graphs
# ==============================================================================
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import style

# Preprocessing and modeling
# ==============================================================================
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Warnings configuration
# ==============================================================================
style.use('ggplot') or plt.style.use('ggplot')

### EXTRACTING DATA INPUT ###
# ==============================================================================
data = None
try:
    df = pd.read_excel('/Users/lorena/Git/PCA/INX_Future_Inc_Employee_Performance_CDS_Project2_Data_V1.8.xls')
except IOError:
    print("Cannot open the xls file")

df = df.drop(columns=['EmpNumber'])

# VISUALIZING DATA
pd.set_option('display.max_columns', None)
df.groupby('EmpDepartment')['PerformanceRating'].mean()
df.groupby('EmpDepartment')['PerformanceRating'].value_counts()

# Creating a new dataframe to analyze each department separately
df_department = pd.get_dummies(df['EmpDepartment'])
print(df_department)
print(type(df_department))

df_performance = df['PerformanceRating']
print(type(df_performance))
print(df_performance)

# Plotting a separate bar graph for performance of each department using seaborn
plt.figure(figsize=(10, 5))
plt.subplot(2, 3, 1)
sns.barplot(x=df_performance, y=df_department['Sales'])
plt.subplot(2, 3, 2)
sns.barplot(x=df_performance, y=df_department['Research & Development'])
plt.subplot(2, 3, 3)
sns.barplot(x=df_performance, y=df_department['Human Resources'])
plt.subplot(2, 3, 4)
sns.barplot(x=df_performance, y=df_department['Data Science'])
plt.subplot(2, 3, 5)
sns.barplot(x=df_performance, y=df_department['Development'])
plt.subplot(2, 3, 6)
sns.barplot(x=df_performance, y=df_department['Finance'])
# plt.show()


### TRANSFORMING CATEGORICAL DATA TO NUMERIC ###
# ==============================================================================

# 1. Select categorical columns (strings) and numerical columns(int, floats)
# ==============================================================================

numerical_columns_selector = selector(dtype_exclude=object)
categorical_columns_selector = selector(dtype_include=object)

numerical_columns = numerical_columns_selector(df)
categorical_columns = categorical_columns_selector(df)

# 2. Create our ColumnTransfomer by specifying three values: the preprocessor name, the transformer, and the columns.
# First, let’s create the preprocessors for the numerical and categorical parts.
# ==============================================================================

categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")
numerical_preprocessor = StandardScaler()

# 3. Create the transformer and associate each of these preprocessors with their respective columns
# ==============================================================================

# NOTE: ColumnsTransformer will internally call fit_transform or transform!
preprocessor = ColumnTransformer([
    ('one-hot-encoder', categorical_preprocessor, categorical_columns),
    ('standard_scaler', numerical_preprocessor, numerical_columns)])

df_prepro = preprocessor.fit_transform(df)

names_columns_preprocessor = preprocessor.get_feature_names_out()  # Column's names already changed in preprocessor step(they aren't X columns anymore)

df_scaled_named = pd.DataFrame(data=df_prepro,
                               columns=names_columns_preprocessor)  # Df with scaled and transformed values + new transformed columns names

### DATA CORRELATION MATRIX ###

corr_matrix = df_scaled_named.corr(method='pearson')

plt.figure(figsize=(16, 6))
heatmap = sns.heatmap(corr_matrix, vmin=-1, vmax=1, annot=True, cmap='BrBG', fmt='.1f')
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize': 18}, pad=12)
# plt.show()


## Here we have selected only the important columns

X = df_scaled_named.iloc[:, [12, 20, 46, 52, 56, 57, 58, 59, 60]]
y = df['PerformanceRating']

#### MODELS ####
# ==============================================================================

# Splitting into train and test for calculating the accuracy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

# Logistic Regression Model
# ==============================================================================

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_predict_log = logreg.predict(X_test)
print(logreg)
print('Logistic Regression Accuracy: {:.3f}'.format(accuracy_score(y_test, y_predict_log)))

# Regression Logistic confusion matrix
print(confusion_matrix(y_test, y_predict_log))

# Support Vector Machine
# ==============================================================================

svc = SVC(kernel='rbf', C=100, random_state=10)
svc.fit(X_train, y_train)
y_predict_svc = svc.predict(X_test)
print('Support Vector Machine Accuracy:{:.3F}'.format(accuracy_score(y_test, y_predict_svc)))

print(confusion_matrix(y_test, y_predict_svc))

# Decision Tree with GridSearchCV
# ==============================================================================
parameters = [{'min_samples_split': [5], 'criterion': ['gini']}]

grid_tree = GridSearchCV(
    estimator=DecisionTreeClassifier(
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=123
    ),
    param_grid=parameters,
    cv=10,
    refit=True,

    return_train_score=True)

grid_tree.fit(X_train, y_train)

## 1. Best ccp_alpha value
# ------------------------------------------------------------------------------
print(grid_tree.best_params_)

## 2. Final tree structure
# ------------------------------------------------------------------------------
final_model = grid_tree.best_estimator_
print(f"Profundidad del árbol: {final_model.get_depth()}")
print(f"Número de nodos terminales: {final_model.get_n_leaves()}")

## 3. Accuracy
# -------------------------------------------------------------------------------

y_predict_dtree = grid_tree.predict(X_test)
accuracy = accuracy_score(
    y_true=y_test,
    y_pred=y_predict_dtree,
    normalize=True
)
print('Decision Tree with GridSearchCV:{:.3F}'.format(accuracy))
print(classification_report(y_test, y_predict_dtree))

# Random Forest with GridSearchCV
# ==============================================================================

parameters_rf = {'min_samples_split': [5], 'criterion': ['gini'], 'min_samples_leaf': [3]}

# Grid search with Cross Validation
# -------------------------------------------------------------------------------
model = RandomForestClassifier(random_state=33, oob_score=True)
grid_rf = GridSearchCV(
    estimator=model,
    param_grid=parameters_rf,
    scoring='accuracy',
    n_jobs=multiprocessing.cpu_count() - 1,
    cv=RepeatedKFold(n_splits=5, n_repeats=3, random_state=123),
    refit=True,
    verbose=0,
    return_train_score=True
)

grid_rf.fit(X=X_train, y=y_train)

# Predicting the model
# -------------------------------------------------------------------------------
y_predict_rf = grid_rf.predict(X_test)
accuracy_rf = accuracy_score(
    y_true=y_test,
    y_pred=y_predict_rf,
    normalize=True
)
print('Random forest with GridSearchCV:{:.3F}'.format(accuracy_rf))
print(classification_report(y_test, y_predict_rf))

# OOB
# -------------------------------------------------------------------------------

model.fit(X_train, y_train)
print('Score training: ', model.score(X_train, y_train))

print('OOB score: ', model.oob_score_)
print('Score test:', model.score(X_test, y_test))

# K-Nearest Neighbor
# ==============================================================================

# 1. GridSearch to extract the number of K
metrics = ['euclidean', 'manhattan']
neighbors = np.arange(1, 16)
param_grid = dict(metric=metrics, n_neighbors=neighbors)

grid_knn = GridSearchCV(estimator=KNeighborsClassifier(),
                        param_grid=param_grid,
                        scoring='accuracy',
                        cv=RepeatedKFold(n_splits=5, n_repeats=3, random_state=123),
                        refit=True)
grid_knn.fit(X_train, y_train)

# since refit=True,we can directly use grid_search object above as our final best model or you can do as follow:
print('best_params is :', grid_knn.best_params_)
print('best_estimator is :', grid_knn.best_estimator_)

# accuracy on test data
print('accuracy_score_knn', accuracy_score(y_test, grid_knn.predict(X_test)))

# Artificial Neural Network
# ==============================================================================

# Training the model
from sklearn.neural_network import MLPClassifier

model_mlp = MLPClassifier(hidden_layer_sizes=(100, 100, 100), batch_size=10, learning_rate_init=0.01, max_iter=2000,
                          random_state=10)
model_mlp.fit(X_train, y_train)

# Predicting the model
y_predict_mlp = model_mlp.predict(X_test)

# Finding accuracy, precision, recall and confusion matrix
print(accuracy_score(y_test, y_predict_mlp))
print(classification_report(y_test, y_predict_mlp))

# XGBoost Classifier
# ==============================================================================
model_XGB = XGBClassifier()
model_XGB.fit(X_train, y_train)

# Predicting the model
y_predict_xgb = model_XGB.predict(X_test)

# Finding accuracy, precision, recall and confusion matrix
print('accuracy_score_XGB', accuracy_score(y_test, y_predict_xgb))
print(classification_report(y_test, y_predict_xgb))
