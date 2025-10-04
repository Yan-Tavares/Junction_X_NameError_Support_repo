from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import os
import sys
import pandas as pd

#########################################################
#Setting up the path for interpreters with relative paths
#########################################################

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(os.path.dirname(current_file_path))
grand_parent_directory = os.path.dirname(parent_directory)

# Add the grad parent directory to the system path
sys.path.append(grand_parent_directory)
# Change the current working directory to the parent directory
os.chdir(grand_parent_directory)

#########################################################
# Perpare datsets
#########################################################
data = pd.read_csv('support files/sample database/classifier_cluster_sample_data.csv')

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


#########################################################
# Naive Bayes classifier
#########################################################
from sklearn.naive_bayes import GaussianNB

NB_clf = GaussianNB()
NB_clf.fit(X_train, y_train)

print('-------------------------------------')
print('Training Bayes classifier')
#########################################################
# SVM
#########################################################
from sklearn.svm import SVC

param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

SVM_clf = SVC(probability=True)
SVM_clf.fit(X_train, y_train)

SVM_clf_GS = GridSearchCV(SVM_clf, param_grid, cv=5)
SVM_clf_GS.fit(X_train, y_train)
SVM_clf = SVM_clf_GS.best_estimator_

print('-------------------------------------')
print('Training SVM with grid search')
print(f"Best parameters:\n{SVM_clf_GS.best_params_}")

#########################################################
# Decision tree
########################################################
from sklearn.tree import DecisionTreeClassifier

param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [1, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [None, 'sqrt', 'log2'],
    'splitter': ['best', 'random']
}

print('-------------------------------------')
print('Training Decision Tree with grid search')

DT_clf = DecisionTreeClassifier()
DT_clf_GS = GridSearchCV(DT_clf, param_grid, cv=5)
DT_clf_GS.fit(X_train, y_train)
DT_clf = DT_clf_GS.best_estimator_

print(f"Best parameters:\n{DT_clf_GS.best_params_}")

########################################################
# Random forest
########################################################
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

rf_param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

print('-------------------------------------')
print('Training Random Forest with grid search')

RF_clf = RandomForestClassifier(random_state=42)
RF_clf_GS = GridSearchCV(RF_clf, rf_param_grid, cv=5)
RF_clf_GS.fit(X_train, y_train)
RF_clf = RF_clf_GS.best_estimator_


print(f"Best parameters:\n{RF_clf_GS.best_params_}")

########################################################
# XGBoost
########################################################
xgb_param_grid = {
    'n_estimators': [25, 50],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0]
}

print('-------------------------------------')
print('Training XGBoost with grid search')

XGB_clf = XGBClassifier(random_state=42)
XGB_clf_GS = GridSearchCV(XGB_clf, xgb_param_grid, cv=5)
XGB_clf_GS.fit(X_train, y_train)
XGB_clf = XGB_clf_GS.best_estimator_


print(f"Best parameters:\n{XGB_clf_GS.best_params_}")

########################################################
# Ensemble model
########################################################

# Ensemble model: average predicted probabilities (soft voting)
proba_SVM = SVM_clf.predict_proba(X_test)
proba_NB = NB_clf.predict_proba(X_test)
proba_DT = DT_clf.predict_proba(X_test)
proba_RF = RF_clf.predict_proba(X_test)
proba_XGB = XGB_clf.predict_proba(X_test)

avg_proba = (proba_SVM + proba_NB + proba_DT + proba_RF + proba_XGB) / 5 # Average probabilities
ensemble_pred = avg_proba.argmax(axis=1) # Classify

#########################################################
# RESULTS
#########################################################
print('-------------------------------------')
print('RESULTS\n')
y_pred_SVM = SVM_clf.predict(X_test)
y_pred_NB = NB_clf.predict(X_test)
y_pred_DT = DT_clf.predict(X_test)
y_pred_RF = RF_clf.predict(X_test)
y_pred_XGB = XGB_clf.predict(X_test)

print(y_pred_XGB)

print("Bayes accuracy:", accuracy_score(y_test, y_pred_NB))
print("SVM Accuracy:", accuracy_score(y_test, y_pred_SVM))
print("DT Accuracy:", accuracy_score(y_test, y_pred_DT))
print("RF Accuracy:", accuracy_score(y_test, y_pred_RF))
print("XGB Accuracy:", accuracy_score(y_test, y_pred_XGB))
print("Ensemble Accuracy:", accuracy_score(y_test, ensemble_pred))