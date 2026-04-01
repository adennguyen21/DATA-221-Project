# KNN model for Heart Failure Dataset by Jacob J
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import pandas as pd

# Convert heart.csv as a dataframe
heart_dataframe = pd.read_csv("heart.csv")

# Split dataset into features and target
matrix_heart_X = heart_dataframe.drop("HeartDisease", axis=1)
target_heart_y = heart_dataframe["HeartDisease"]

#train test split
X_train, X_test, y_train, y_test = train_test_split(
    matrix_heart_X,
    target_heart_y,
    test_size=0.3,
    random_state=42
)

# One-hot encode them separately
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

# Scale features
scaler = StandardScaler()
scaler.fit_transform(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create KNN model
number_of_neighbors = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
for k in number_of_neighbors:
    knn_model = KNeighborsClassifier(n_neighbors=k, metric="euclidean")

    # Train model
    knn_model.fit(X_train, y_train)

    # Predict on test data
    y_pred = knn_model.predict(X_test)

    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"k value: {k}\n")

    # Confusion Matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print(f"Accuracy: {accuracy} Precision: {precision} Recall: {recall} F1 Score: {f1}\n")


