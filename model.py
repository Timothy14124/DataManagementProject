import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score

df = pd.read_csv("cleaned_data.csv")

monthly_total = (df.groupby(["AREA NAME", "Year", "Month"]).size().reset_index(name="CrimeCount"))
top_crimes = (df.groupby(["AREA NAME", "Year", "Month"])["Crm Cd Desc"].agg(lambda x: x.mode()[0]).reset_index(name="MostLikelyCrime"))
monthly_data = pd.merge(monthly_total, top_crimes,on=["AREA NAME", "Year", "Month"])
monthly_data = monthly_data[monthly_data["CrimeCount"] > 50]

monthly_data = monthly_data.sort_values(by=["AREA NAME", "Year", "Month"])
monthly_data["PrevCrimeCount"] = (monthly_data.groupby("AREA NAME")["CrimeCount"].shift(1))
monthly_data = monthly_data.dropna()

monthly_data["AreaCode"] = (monthly_data["AREA NAME"].astype("category").cat.codes)

#Crime count predictor
X = monthly_data[["AreaCode","Year","Month","PrevCrimeCount"]]
y_crime = monthly_data["CrimeCount"]
X_train, X_test, y_train, y_test = train_test_split(X, y_crime, test_size=0.2, random_state=42)
crime_model = RandomForestRegressor(n_estimators=100, random_state=42)
crime_model.fit(X_train, y_train)
crime_preds = crime_model.predict(X_test)
#print("Crime count predictor evaluation:")
#print("MAE:", mean_absolute_error(y_test, crime_preds))
#print("R^2:", r2_score(y_test, crime_preds))

#Most likely crime predictor
y_crime_type = monthly_data["MostLikelyCrime"]
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y_crime_type, test_size=0.2, random_state=42)
crime_type_model = RandomForestClassifier(n_estimators=100, random_state=42)
crime_type_model.fit(X_train2, y_train2)
crime_type_preds = crime_type_model.predict(X_test2)
#print("Most likely crime predictor evaluation:")
#print("Accuracy score:", accuracy_score(y_test2, crime_type_preds))

#Create safety scores
area_avg = (monthly_data.groupby("AREA NAME")["CrimeCount"].mean().reset_index())
area_avg.rename(columns={"CrimeCount": "AvgMonthlyCrimes"},inplace=True)
min_crime = area_avg["AvgMonthlyCrimes"].min()
max_crime = area_avg["AvgMonthlyCrimes"].max()
area_avg["SafetyScore"] = (100 -((area_avg["AvgMonthlyCrimes"] - min_crime) / (max_crime - min_crime)) * 100)
area_avg["SafetyScore"] = (area_avg["SafetyScore"].round(2))

#Evaluations for models
with open("evaluation.txt", "w") as f:
    print("Crime count predictor evaluation:", file=f)
    print("MAE:", mean_absolute_error(y_test, crime_preds), file=f)
    print("R^2:", r2_score(y_test, crime_preds), file=f)
    print("Most likely crime predictor evaluation:", file=f)
    print("Accuracy score:", accuracy_score(y_test2, crime_type_preds), file=f)

#Prediction for May 2026
latest_data = monthly_data[(monthly_data["Year"] == 2024) & (monthly_data["Month"] == 12)]
temp = pd.DataFrame()
temp["AREA NAME"] = latest_data["AREA NAME"]
temp["AreaCode"] = latest_data["AreaCode"]
temp["Year"] = 2026
temp["Month"] = 5
temp["PrevCrimeCount"] = latest_data["CrimeCount"]
X_4 = temp[["AreaCode", "Year", "Month", "PrevCrimeCount"]]
temp["PredictedCrimeCount"] = (crime_model.predict(X_4)).round()
temp["PredictedCrimeType"] = (crime_type_model.predict(X_4))
final_predictions = temp[["AREA NAME", "PredictedCrimeCount", "PredictedCrimeType"]]
final_predictions = final_predictions.merge(area_avg[["AREA NAME", "SafetyScore"]], on="AREA NAME",how="left")

final_predictions.to_csv("final_predictions.csv", index=False)