from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load heart disease data
heart_data = pd.read_csv("heart.csv")

# Separate feature matrix data and label vector
X = heart_dataframe.drop("HeartDisease", axis=1)
y = heart_dataframe["HeartDisease"]

# Split data into 70% training and 30% testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# One-hot encode data
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

# Scale feature data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Create logistic regression model and fit to training set
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

# Test model
y_pred = logistic_model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
confusion_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:\n", confusion_matrix)