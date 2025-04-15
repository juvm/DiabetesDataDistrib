import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from collections import Counter


df = pd.read_csv("../archive/diabetes_012_health_indicators_BRFSS2015.csv")

# Count the number of instances for each class in the target column
target_column = "Diabetes_012"
class_counts = df[target_column].value_counts()
print(class_counts)

# 0.0    213703    84.2%
# 2.0     35346    13.9%
# 1.0      4631    1.9%
# Name: Diabetes_012, dtype: int64

# IMPLEMENTATION OF THE RANDOM FOREST ALGORITHM
# Separate features and target
X = df.drop(columns=["Diabetes_012"])  # Assuming "Diabetes_012" is the target
y = df["Diabetes_012"]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

X_sampled = X.sample(n=100000, random_state=42)  # Use only 10K samples
y_sampled = y[X_sampled.index]

smote_tomek = SMOTETomek(
    sampling_strategy='auto',
    smote=SMOTE(k_neighbors=3),  # Pass SMOTE with parameters here
    random_state=42,
    n_jobs=-1
)
#X_resampled, y_resampled = smote_tomek.fit_resample(X, y)

X_resampled, y_resampled = smote_tomek.fit_resample(X_sampled, y_sampled)
print("Resampling successful!")
print("abc3")

# Print the new class distribution
print("Original class distribution:", Counter(y)) 
print("New class distribution:", Counter(y_resampled)) #8460, 8448, 8448


# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.25, random_state=42)

rf = RandomForestClassifier(
    n_estimators=30,        # 300 trees
    min_samples_split=15,    # Minimum 15 samples required to split an internal node
    min_samples_leaf=6,     # Minimum 15 samples required to be a leaf node
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
