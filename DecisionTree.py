import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load data into a dataframe
heart_dataframe = pd.read_csv("heart.csv")

# Split the data first
matrix_heart_X = heart_dataframe.drop("HeartDisease", axis=1)
target_heart_Y = heart_dataframe["HeartDisease"]

# Train-Test-Split
X_train, X_test, y_train, y_test = train_test_split(
    matrix_heart_X, target_heart_Y, test_size=0.2, random_state=42
)

# One-hot encode them separately
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

# Align the columns
# Ensures that both datasets have the same columns
X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

# Build the model
model = DecisionTreeClassifier(
    criterion = 'entropy',
    max_depth=4,
    min_samples_leaf=3,
    min_samples_split=5,
    random_state=42
)

model.fit(X_train, y_train)

# Predict
y_predict = model.predict(X_test)

# Evaluate and display the evaluations
print("Accuracy:", accuracy_score(y_test, y_predict))
print("Precision:", precision_score(y_test, y_predict))
print("Recall:", recall_score(y_test, y_predict))
print("F1-score:", f1_score(y_test, y_predict))

cm = confusion_matrix(y_test, y_predict)
print("Confusion Matrix:")
print("TN FP")
print("FN TP")
print(cm)