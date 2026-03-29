# KNN model for Heart Failure Dataset by Jacob J

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import pandas as pd

# Convert heart.csv as a dataframe
heart_dataframe = pd.read_csv("heart.csv")

# Separate categorical columns
categorical_columns = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]

# One-hot encode
heart_df_encoded = pd.get_dummies(heart_dataframe, columns=categorical_columns, drop_first=False)

# Split dataset into features and target
matrix_heart_X = heart_df_encoded.drop("HeartDisease", axis=1)
target_heart_y = heart_df_encoded["HeartDisease"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(matrix_heart_X)

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    target_heart_y,
    test_size=0.3,
    random_state=42
)

# Create KNN model
knn_model = KNeighborsClassifier(n_neighbors=1)

# Train model
knn_model.fit(X_train, y_train)

# Predict on test data
y_pred = knn_model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
