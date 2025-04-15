"""import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Example class distribution (replace with your actual class counts)
class_labels = ['No Diabetes', 'Prediabetes', 'Diabetes']
class_counts = [213703, 4631, 35346]  # Replace with actual numbers

# Define colors
colors = ['gray', 'black', 'blue']

# Create bar plot
plt.figure(figsize=(6, 4))
sns.barplot(x=class_labels, y=class_counts, palette=colors)

# Add labels
plt.xlabel('Category')
plt.ylabel('Number of Samples')
plt.title('Class Distribution')

# Show values on bars
for i, count in enumerate(class_counts):
    plt.text(i, count + 10, str(count), ha='center', fontsize=12)

plt.show()"""





"""
import pandas as pd

# Load the dataset
df = pd.read_csv("../archive/diabetes_012_health_indicators_BRFSS2015.csv")

# Select all 21 features (excluding the target variable if needed)
features = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke", "HeartDiseaseorAttack",
    "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare",
    "NoDocbcCost", "GenHlth", "MentHlth", "PhysHlth", "DiffWalk", "Sex",
    "Age", "Education", "Income"
]

# Compute summary statistics
summary_table = df[features].describe().T  # Transpose for better readability

# Rename index and columns for clarity
summary_table.index.name = "Feature"
summary_table.columns = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]

# Display the table
print(summary_table)
"""








"""

import pandas as pd
# import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("../archive/diabetes_012_health_indicators_BRFSS2015.csv")

# -----------------------------------------------------------------------------------------------------------

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












# 84.18%
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
df = pd.read_csv("../archive/diabetes_012_health_indicators_BRFSS2015.csv")

# Separate features and target
X = df.drop(columns=["Diabetes_012"])  # Assuming "Diabetes_012" is the target
y = df["Diabetes_012"]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.6f}")