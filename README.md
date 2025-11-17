import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
df = pd.read_csv(r"C:\Users\acer\Downloads\archive(1)\Titanic-Dataset.csv")
print(df.head())
print(df.info())
Features = ["Pclass","SibSp","Parch","Fare"]
df = df[Features + ["Survived"]] 
print(df.head())
X = df[Features]
Y = df["Survived"]
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.3,random_state = 42)
dt = DecisionTreeClassifier(criterion = "entropy", max_depth = 3,random_state = 42)
dt.fit(X_train, Y_train)
Y_pred = dt.predict(X_test)
print(Y_pred)
print(f"The accuracy of the dataset is {accuracy_score(Y_test, Y_pred)}")
print(f"The classification report of the dataset is {classification_report(Y_test, Y_pred)}")
print(f"The confusion matrix of the dataset is {confusion_matrix(Y_test, Y_pred)}")
plt.figure(figsize=(12,8))
plot_tree(dt,feature_names = X.columns,class_names = ["Not Survived","Survived"],filled = True,rounded = True)
plt.show()
