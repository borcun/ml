import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

def viewCorrelationHeatMap(data, plt, title) :
    sns.heatmap(data.corr(), annot = True, linewidths = 1, cbar = True, cmap = "ocean_r", fmt = ".2f")
    plt.suptitle(title, fontsize = 16)
    plt.show()

def executeXGBoost(data, year):
    xgb = XGBClassifier(random_state = 42, n_jobs = 1, max_depth = 5, subsample = 0.5, n_estimators = 100)
    columns = ["Economy (GDP per Capita)", "GDP per capita", "Economy..GDP.per.Capita.",
               "Family", "Social support",
               "Health (Life Expectancy)", "Healthy life expectancy", "Health..Life.Expectancy.",
               "Freedom", "Freedom to make life choices",
               "Generosity",
               "Trust (Government Corruption)", "Perceptions of corruption", "Trust..Government.Corruption."]
    
    for col in columns:
        if col in data:
            y = data[col].astype('int')
            x = data.drop(col, axis = 1)
    
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)
        
            xgb.fit(x_train, y_train)
            y_pred = xgb.predict(x_test)
            score = accuracy_score(y_test, y_pred)
            
            print(col)
            print("XGBoost Accuracy Score " + col + " in " + str(year), score)
            print(classification_report(y_test, y_pred))
            
            conf_mat = confusion_matrix(y_test, y_pred)
            sns.heatmap(conf_mat, annot = True, linewidths = 1, cbar = True, cmap = "ocean_r", fmt = "g")
            plt.xlabel("Y Prediction")
            plt.ylabel("Y Test")
            plt.title("XGBoost Confusion Matrix for " + col + " in " + str(year))
            plt.show()

def main() :
    warnings.filterwarnings("ignore")
    plt.figure(figsize = (16, 10))
    years = range(2015, 2020)
    
    for year in years:
        data = pd.read_csv('../archive/' + str(year) + '.csv')
        ref_data = data.select_dtypes(include=["float64"]).copy()
        viewCorrelationHeatMap(ref_data, plt, "Correlation Map in " + str(year))

        lbe = LabelEncoder()

        if "Country or region" in data:
            data["Country or region"] = lbe.fit_transform(data["Country or region"])

        if "Country" in data:
            data["Country"] = lbe.fit_transform(data["Country"])

        if "Region" in data:
            data["Region"] = lbe.fit_transform(data["Region"])
            
        executeXGBoost(data, year)
    
if __name__ == "__main__":
    main()
