import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.utils import resample
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# -----------------------------------------------------------------------------------------------------------
# -------------------------- 0. RATIO, original : 84.9448% || 85.0473% -----------------------------
"""
df = pd.read_csv("../archive/diabetes_012_health_indicators_BRFSS2015.csv")

# Count the number of instances for each class in the target column
target_column = "Diabetes_012"
class_counts = df[target_column].value_counts()
print(class_counts)

# 0.0    213703    84.2%
# 2.0     35346    13.9%
# 1.0      4631    1.9%
# Name: Diabetes_012, dtype: int64

# Separate features and target
X = df.drop(columns=["Diabetes_012"])  # Assuming "Diabetes_012" is the target
y = df["Diabetes_012"]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

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

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.6f}")

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')  # For multi-class classification
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Print the results
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
"""
# -----------------------------------------------------------------------------------------------------------











# -----------------------------------------------------------------------------------------------------------
# -------------------------- 1. RATIO, /10 : 85.1466% || 85.0363% -----------------------------
"""
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

# Define target dataset size (10x smaller)
target_size = len(y) // 10  

# Downsample dataset while keeping class ratios
X_resampled, y_resampled = resample(
    X, y, replace=False, n_samples=target_size, stratify=y, random_state=42
)

# Check class distribution
print(pd.Series(y_resampled).value_counts(normalize=True))  # Should match original ratios
print(pd.Series(y_resampled).value_counts())  # Absolute counts

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

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.6f}")

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')  # For multi-class classification
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Print the results
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
"""
# -----------------------------------------------------------------------------------------------------------




# -----------------------------------------------------------------------------------------------------------
# -------------------------- 2. RATIO, *2 : 85.9484% || 85.7695% -----------------------------
"""
df = pd.read_csv("../archive/diabetes_012_health_indicators_BRFSS2015.csv")

# Count the number of instances for each class in the target column
target_column = "Diabetes_012"
class_counts = df[target_column].value_counts()
print(class_counts)

# IMPLEMENTATION OF THE RANDOM FOREST ALGORITHM
# Separate features and target
X = df.drop(columns=["Diabetes_012"])  # Assuming "Diabetes_012" is the target
y = df["Diabetes_012"]

# Calculate new dataset size (2x original)
target_size = len(y) * 2  

# Compute new class distribution while maintaining ratio
class_ratios = y.value_counts(normalize=True).to_dict()
sampling_strategy = {cls: int(ratio * target_size) for cls, ratio in class_ratios.items()}

# Apply SMOTE with proportional class distribution
smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Check new class distribution
print(pd.Series(y_resampled).value_counts(normalize=True))  # Should be balanced
print(pd.Series(y_resampled).value_counts())  # Absolute counts (4,631 per class)

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

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.6f}")

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')  # For multi-class classification
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Print the results
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
"""
# -----------------------------------------------------------------------------------------------------------







# -----------------------------------------------------------------------------------------------------------
# -------------------------- 3. EQUAL, 4631/4631/4631 : 51.8998% || 52.4180%-----------------------------
"""
df = pd.read_csv("../archive/diabetes_012_health_indicators_BRFSS2015.csv")

# Count the number of instances for each class in the target column
target_column = "Diabetes_012"
class_counts = df[target_column].value_counts()
print(class_counts)

# IMPLEMENTATION OF THE RANDOM FOREST ALGORITHM
# Separate features and target
X = df.drop(columns=["Diabetes_012"])  # Assuming "Diabetes_012" is the target
y = df["Diabetes_012"]

# Define target class size (middle class count)
target_size = 4631  

# Apply RandomUnderSampler to match all classes to 4,631 instances
undersample = RandomUnderSampler(sampling_strategy={0.0: target_size, 2.0: target_size}, random_state=42)
X_resampled, y_resampled = undersample.fit_resample(X, y)

# Check new class distribution
print(pd.Series(y_resampled).value_counts(normalize=True))  # Should be balanced
print(pd.Series(y_resampled).value_counts())  # Absolute counts (4,631 per class)

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

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.6f}")

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')  # For multi-class classification
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Print the results
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
"""
# -----------------------------------------------------------------------------------------------------------






# -----------------------------------------------------------------------------------------------------------
# -------------------------- 4. EQUAL, 35346/35346/35346 : 78.5477% || 78.221%-----------------------------
"""
df = pd.read_csv("../archive/diabetes_012_health_indicators_BRFSS2015.csv")

# Count the number of instances for each class in the target column
target_column = "Diabetes_012"
class_counts = df[target_column].value_counts()
print(class_counts)

# IMPLEMENTATION OF THE RANDOM FOREST ALGORITHM
# Separate features and target
X = df.drop(columns=["Diabetes_012"])  # Assuming "Diabetes_012" is the target
y = df["Diabetes_012"]

# First, undersample class 0.0 to 35,346
undersample = RandomUnderSampler(sampling_strategy={0.0: 35346}, random_state=42)
X_under, y_under = undersample.fit_resample(X, y)

# Now, oversample class 1.0 to 35,346 using SMOTE
smote = SMOTE(sampling_strategy={1.0: 35346}, random_state=42, k_neighbors=5)
X_resampled, y_resampled = smote.fit_resample(X_under, y_under)

# Check new class distribution
print(pd.Series(y_resampled).value_counts(normalize=True))  # Should be balanced
print(pd.Series(y_resampled).value_counts())  # Absolute counts (4,631 per class)

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

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.6f}")

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')  # For multi-class classification
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Print the results
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
"""
# -----------------------------------------------------------------------------------------------------------






# randomoversampler ROS --> duplicates instances --> overfitting, 
# vs SMOTE --> KNN --> diversity
# -----------------------------------------------------------------------------------------------------------
# -------------------------- 5. ROS EQUAL, TOTAL/3 : 82.5307% || 79.3504%-----------------------------
"""
df = pd.read_csv("../archive/diabetes_012_health_indicators_BRFSS2015.csv")

# Count the number of instances for each class in the target column
target_column = "Diabetes_012"
class_counts = df[target_column].value_counts()
print(class_counts)

# IMPLEMENTATION OF THE RANDOM FOREST ALGORITHM
# Separate features and target
X = df.drop(columns=["Diabetes_012"])  # Assuming "Diabetes_012" is the target
y = df["Diabetes_012"]

target_size=84560

# First, undersample the majority class (0.0) to 84,560
undersample = RandomUnderSampler(sampling_strategy={0.0: target_size}, random_state=42)
X_resampled, y_resampled = undersample.fit_resample(X, y)

# Then, oversample the minority and middle classes (1.0 and 2.0) to 84,560
over_sample = RandomOverSampler(sampling_strategy={1.0: target_size, 2.0: target_size}, random_state=42)
X_resampled, y_resampled = over_sample.fit_resample(X_resampled, y_resampled)

# Check new class distribution
print(pd.Series(y_resampled).value_counts(normalize=True))  # Should be balanced
print(pd.Series(y_resampled).value_counts())  # Absolute counts (4,631 per class)

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

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.6f}")
"""
# -----------------------------------------------------------------------------------------------------------









# -----------------------------------------------------------------------------------------------------------
# -------------------------- 6. SMOTE EQUAL, TOTAL/3 : 78.8994% || 77.0829% -----------------------------

df = pd.read_csv("../archive/diabetes_012_health_indicators_BRFSS2015.csv")

# Count the number of instances for each class in the target column
target_column = "Diabetes_012"
class_counts = df[target_column].value_counts()
print(class_counts)

# IMPLEMENTATION OF THE RANDOM FOREST ALGORITHM
# Separate features and target
X = df.drop(columns=["Diabetes_012"])  # Assuming "Diabetes_012" is the target
y = df["Diabetes_012"]

# First, undersample class 0.0 to 84,560
undersample = RandomUnderSampler(sampling_strategy={0.0: 84560}, random_state=42)
X_under, y_under = undersample.fit_resample(X, y)

# Now, oversample class 0.0 and 1.0 to 84,560 using SMOTE
smote = SMOTE(sampling_strategy={1.0: 84560, 2.0: 84560}, random_state=42, k_neighbors=5)

# Apply SMOTE
X_resampled, y_resampled = smote.fit_resample(X_under, y_under)


# Check new class distribution
print(pd.Series(y_resampled).value_counts(normalize=True))  # Should be balanced
print(pd.Series(y_resampled).value_counts())  # Absolute counts (4,631 per class)

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

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.6f}")

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')  # For multi-class classification
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Print the results
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
# -----------------------------------------------------------------------------------------------------------






# -----------------------------------------------------------------------------------------------------------
# -------------------------- 7. EQUAL, 8460/8451/8451 : 93.3065% || 91.5773%-----------------------------
"""
df = pd.read_csv("../archive/diabetes_012_health_indicators_BRFSS2015.csv")

# Count the number of instances for each class in the target column
target_column = "Diabetes_012"
class_counts = df[target_column].value_counts()
print(class_counts)

# IMPLEMENTATION OF THE RANDOM FOREST ALGORITHM
# Separate features and target
X = df.drop(columns=["Diabetes_012"])  # Assuming "Diabetes_012" is the target
y = df["Diabetes_012"]

X_sampled = X.sample(n=10000, random_state=42)  # Use only 10K samples
y_sampled = y[X_sampled.index]

smote_tomek = SMOTETomek(
    sampling_strategy='auto',
    smote=SMOTE(k_neighbors=3),  # Pass SMOTE with parameters here
    random_state=42,
    n_jobs=-1
)

X_resampled, y_resampled = smote_tomek.fit_resample(X_sampled, y_sampled)
print("Resampling successful!")

# Print the new class distribution
print("New class distribution:", Counter(y_resampled)) #8460, 8451, 8451

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

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.6f}")
"""
# -----------------------------------------------------------------------------------------------------------
