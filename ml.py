import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# total data count in a file
TOP_LIMIT = 100

# data suffix list
years = range(2011, 2023)
dataset = {}
top_list = {}

def executeModel(model, name):
  annual_list = top_list[2011]
  x = annual_list.drop(["location"], axis = 1)
  y = annual_list["location"]
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

  model.fit(x_train, y_train)
  y_pred = model.predict(x_test)
  score = accuracy_score(y_test, y_pred)
  print(name, "model accuracy:", score)

  conf_mat = confusion_matrix(y_test, y_pred)
  sns.heatmap(conf_mat, annot=True, cbar=False, fmt="g")
  plt.xlabel("y-pred")
  plt.ylabel("y-test")
  plt.title(name)
  print(classification_report(y_test, y_pred))
  

def main():
  warnings.filterwarnings("ignore")

  encoder = LabelEncoder()
 
  for year in years:
    path = "./dataset/" + str(year) + "_rankings.csv"
    dataset[year] = pd.read_csv(path)
    # filter top elements to copy top list from entire dataset
    top_list[year] = dataset[year].head(TOP_LIMIT)

  top_list[years[0]].info()

  # convert object types to countable enumerations
  for year in years:
    annual_list = top_list[year]
    categoric = annual_list.select_dtypes(include=["object"]).copy()
  
    for j in categoric.columns:
      annual_list[j] = encoder.fit_transform(annual_list[j])

  # show fit data as bar chart
  for year in years:
    plt.figure(figsize=(20,6))
    sns.countplot(x = top_list[year]["location"], data = top_list[year])
    plt.show()

  executeModel(LogisticRegression(), "Logistic Regression")
  executeModel(SVC(), "SVM")
  executeModel(KNeighborsClassifier(), "KNN")
  executeModel(DecisionTreeClassifier(), "Decision Tree")

if __name__ == "__main__":
  main()
