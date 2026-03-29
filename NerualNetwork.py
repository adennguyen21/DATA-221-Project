# Neural Network model for Heart Failure Dataset, By Aden N

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
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

