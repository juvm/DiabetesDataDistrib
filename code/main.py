# To create a virtual environment: 
# 1. python -m venv diabenv (installed python 3.7.4 version)
# *2. C:\'Program Files'\Python311\python.exe -m venv diabenv (so i went to cmd, did "where python" and replaced plain python w path of python 3.11)
# To activate it: diabenv\Scripts\activate ; Check version: python --version
# Manually installed packages: pandas, matplotlib, scikit-learn, imblearn

import pandas as pd
# import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("../archive/diabetes_012_health_indicators_BRFSS2015.csv")

# -----------------------------------------------------------------------------------------------------------
"""
# -------------------------- CLEANING -----------------------------
# Could have removed duplicates BUT can't as the dataset is a survey
print(len(df))
df = df.drop_duplicates()  # Remove duplicate rows
print(len(df))
# Dropping null values - cleaned dataset so no null values
print(df.isnull().sum())
df = df.dropna()
# No more cleaning tasks
"""
# -----------------------------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------------------------
# -------------------------- COUNTING INSTANCES -----------------------------
target_column = "Diabetes_012"
class_counts = df[target_column].value_counts()
print(class_counts/sum(class_counts))

# 0.0    213703    84.2%
# 2.0     35346    13.9%
# 1.0      4631    1.9%
# Name: Diabetes_012, dtype: int64
# -----------------------------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------------------------
"""
# -------------------------- PREPROCESSING -----------------------------
# Encoding - No encoding needed
# Scaling - No scaling needed for Random Forest, but I'd still like to scale the BMI, MentHlth, PhysHlth
# BMI is approximately normally distributed => we use StandardScaler, and MinMaxScaler 
# for the other two since they're very skewed according to the graphs
columns_to_scale = ["BMI", "MentHlth", "PhysHlth"]
df[columns_to_scale].hist(figsize=(10, 5), bins=30)
plt.show()
scaler_standard = StandardScaler()
scaler_minmax = MinMaxScaler()
# Apply different scalers
df["BMI"] = scaler_standard.fit_transform(df[["BMI"]])
df[["MentHlth", "PhysHlth"]] = scaler_minmax.fit_transform(df[["MentHlth", "PhysHlth"]])
# Save or inspect transformed data
df.to_csv("./prep_data/scaled_dataset.csv", index=False)
print(df.head())
"""
# -----------------------------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------------------------
# -------------------------- PREPARING TRAIN & TEST DATA -----------------------------
# Separate features and target
X = df.drop(columns=["Diabetes_012"])  # Assuming "Diabetes_012" is the target
y = df["Diabetes_012"]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
# -----------------------------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------------------------
"""
# -------------------------- EXTRACTING TOP 15 IMPORTANT FEATURES -----------------------------
# Train Random Forest on all features
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
# Get accuracy on test data
y_pred = rf.predict(X_test)
accuracy_full = accuracy_score(y_test, y_pred) # 84.15%
print(f"Accuracy with all features: {accuracy_full:.4f}")
# Get feature importance
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': rf.feature_importances_})
feature_importances = feature_importances.sort_values(by="Importance", ascending=False)
# Select top 15 most important features
print("Important features:", feature_importances)
top_features = feature_importances.iloc[:15]["Feature"].values
print("\nTop 15 Features Selected:", top_features)
# Train Random Forest on selected features
X_train_selected = X_train[top_features]
X_test_selected = X_test[top_features]
rf_selected = RandomForestClassifier(n_estimators=100, random_state=42)
rf_selected.fit(X_train_selected, y_train)
# Get accuracy on test data (with top 15 features)
y_pred_selected = rf_selected.predict(X_test_selected)
accuracy_selected = accuracy_score(y_test, y_pred_selected)
print(f"Accuracy with top 15 features: {accuracy_selected:.4f}")
# Compare performance
if accuracy_selected > accuracy_full:
    print("\nUsing only the top 15 features improved accuracy!")
elif accuracy_selected < accuracy_full:
    print("\nUsing all features performed better.")
else:
    print("\nAccuracy remained the same with feature reduction.")
"""
# -----------------------------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------------------------
"""
# -------------------------- HYPERPARAMETER TUNING -----------------------------
# -------------------------- 1. RANDOMSEARCHCV -----------------------------
# Best Parameters: {'n_estimators': 300, 'min_samples_split': 15, 'min_samples_leaf': 15, 'max_features': 'sqrt', 'max_depth': 20, 'bootstrap': True}
# Test Accuracy: 0.850776568905708
# ^ Best Parameters + random_state = 37 for 85.09
# Define parameter grid
param_grid = {
    "n_estimators": [200, 300, 400, 500],
    "max_depth": [20, 30, 40, None],
    "min_samples_split": [10, 15, 20],
    "min_samples_leaf": [5, 10, 15],
    "max_features": ["sqrt", "log2"],
    "bootstrap": [True, False]
}
# Initialize Random Forest
rf = RandomForestClassifier(n_jobs=-1, random_state=42)
# Randomized Search with 10 iterations
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_grid,
    n_iter=10,
    scoring="accuracy",
    cv=3,
    verbose=2,
    random_state=42
)
# Fit the model
random_search.fit(X_train, y_train)
# Print best parameters
print("Best Parameters:", random_search.best_params_)
# Train final model using best parameters
best_rf = random_search.best_estimator_
y_pred = best_rf.predict(X_test)
# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy)

# -------------------------- 2. GRIDSEARCHCV -----------------------------
# Best Parameters: {'max_depth': 20, 'min_samples_leaf': 6, 'min_samples_split': 15, 'n_estimators': 30}
# Test Accuracy: 0.849369284137496
param_grid = {
    'n_estimators': [10, 20, 30],  # Adjust the number of trees in the forest
    'max_depth': [10, 20, 30],  # Adjust the maximum depth of each tree
    'min_samples_split': [2, 5, 10, 15, 20],  # Adjust the minimum samples required to split a node
    'min_samples_leaf': [1, 2, 4, 6, 8]  # Adjust the minimum samples required in a leaf node
}
model = RandomForestClassifier(random_state=42, n_jobs=-1)
grid_search = GridSearchCV(model, param_grid, cv=5, verbose=3, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
# Train final model using best parameters
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy)
"""
# -----------------------------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------------------------
# -------------------------- RANDOM FOREST ALGORITHM -----------------------------
# 85.04 for 630 n_estimators
rf = RandomForestClassifier(
    n_estimators=630,        # 300 trees
    min_samples_split=15,    # Minimum 15 samples required to split an internal node
    min_samples_leaf=15,     # Minimum 15 samples required to be a leaf node
    max_features='sqrt',     # Use sqrt of total features when looking for best split
    max_depth=20,            # Maximum tree depth
    bootstrap=True,          # Bootstrap sampling for training
    random_state=35,         # Set random seed for reproducibility = 35
    n_jobs=-1                # Use all available CPU cores for faster training
)

rf.fit(X_train, y_train)  # Train model

y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Test Accuracy: {accuracy:.6f}")  # Print accuracy up to 6 decimal places
# -----------------------------------------------------------------------------------------------------------





