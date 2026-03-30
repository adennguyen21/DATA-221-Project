# Neural Network model for Heart Failure Dataset, By Aden N
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import os
import random
import numpy as np
os.environ["PYTHONHASHSEED"] = "42"
random.seed(42)
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, Dropout
from tensorflow.keras.optimizers import Adam
import pandas as pd

# Convert heart.csv as a dataframe
heart_dataframe = pd.read_csv("heart.csv")

# Separate categorical columns
categorical_columns = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]

# One-hot encode
heart_df_encoded = pd.get_dummies(heart_dataframe, columns=categorical_columns, drop_first=False)

# Split dataset
matrix_heart_X = heart_df_encoded.drop("HeartDisease", axis = 1)
target_heart_y = heart_df_encoded["HeartDisease"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(matrix_heart_X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, target_heart_y, test_size=0.3, random_state=42)

# Build model
neural_network_model = Sequential()
neural_network_model.add(InputLayer(input_shape=(X_train.shape[1],)))
neural_network_model.add(Dense(32, activation='relu'))
neural_network_model.add(Dropout(0.2))
neural_network_model.add(Dense(16, activation='relu'))
neural_network_model.add(Dropout(0.1))
neural_network_model.add(Dense(1, activation='sigmoid'))

neural_network_model.compile(loss = "binary_crossentropy", optimizer = Adam(learning_rate = 0.001), metrics = ["accuracy"])

history = neural_network_model.fit(X_train, y_train, epochs = 50, batch_size = 32)

# Evaluate model
y_predicted = (neural_network_model.predict(X_test) > 0.5).astype(int)

# Accuracy
accuracy_score = accuracy_score(y_test, y_predicted)
print("Accuracy Score: ", accuracy_score)

# Precision
precision_score = precision_score(y_test, y_predicted)
print("Precision Score: ", precision_score)

# Recall
recall_score = recall_score(y_test, y_predicted)
print("Recall Score: ", recall_score)

# F1-score
f1_score = f1_score(y_test, y_predicted)
print("F1 Score: ", f1_score)

# Confusion Matrix
confusion_matrix = confusion_matrix(y_test, y_predicted)
print("Confusion Matrix: \n", confusion_matrix)
