# Neural Network model for Heart Failure Dataset, By Aden N

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
import tensorflow as tf
import pandas as pd

# Convert heart.csv as a dataframe
heart_dataframe = pd.read_csv("heart.csv")

# Separate categorical columns
categorical_columns = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]

# One-hot encode
heart_df_encoded = pd.get_dummies(heart_dataframe, columns=categorical_columns, drop_first=False)




