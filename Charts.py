import matplotlib.pyplot as plt
import numpy as np

accuracy_dictionary = {"Logistic Regression": 0.880, "KNN": 0.899, "Decision Tree": 0.841, "Neural Network": 0.870}
precision_dictionary = {"Logistic Regression": 0.917, "KNN": 0.925, "Decision Tree": 0.857, "Neural Network": 0.905}
recall_dictionary = {"Logistic Regression": 0.878, "KNN": 0.902, "Decision Tree": 0.878, "Neural Network": 0.872}
f1_dictionary = {"Logistic Regression": 0.897, "KNN": 0.914, "Decision Tree": 0.867, "Neural Network": 0.888}

models = sorted(accuracy_dictionary.keys())

accuracy = [accuracy_dictionary[k] for k in models]

precision = [precision_dictionary[k] for k in models]

recall = [recall_dictionary[k] for k in models]

f1 = [f1_dictionary[k] for k in models]

# Create the chart
width = 0.2
x = np.arange(len(models))

plt.figure(figsize=(12,6))
plt.bar(x - width*1.5, accuracy, width, label='Accuracy')
plt.bar(x - width*0.5, precision, width, label='Precision')
plt.bar(x + width*0.5, recall, width, label='Recall')
plt.bar(x + width*1.5, f1, width, label='F1 Score')

plt.xticks(x, models, rotation=20)
plt.ylabel("Score")
plt.title("Model Performance Comparison")
plt.legend()
plt.tight_layout()
plt.show()
