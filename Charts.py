import matplotlib.pyplot as plt

accuracy_dictionary = {"Logistic Regression": 0.880, "KNN": 0.899, "Decision Tree": 0.841, "Neural Network": 0.870}
precision_dictionary = {"Logistic Regression": 0.917, "KNN": 0.925, "Decision Tree": 0.857, "Neural Network": 0.905}
recall_dictionary = {"Logistic Regression": 0.878, "KNN": 0.902, "Decision Tree": 0.878, "Neural Network": 0.872}
f1_dictionary = {"Logistic Regression": 0.897, "KNN": 0.914, "Decision Tree": 0.867, "Neural Network": 0.888}

x_accuracy = sorted(accuracy_dictionary.keys())
y_accuracy = [accuracy_dictionary[k] for k in x_accuracy]

x_precision = sorted(precision_dictionary.keys())
y_precision = [precision_dictionary[k] for k in x_accuracy]

x_recall = sorted(recall_dictionary.keys())
y_recall = [recall_dictionary[k] for k in x_accuracy]

x_f1 = sorted(f1_dictionary.keys())
y_f1 = [f1_dictionary[k] for k in x_accuracy]

