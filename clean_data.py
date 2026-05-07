import pandas as pd
import numpy as np

df = pd.read_csv("Crime_Data_from_2020_to_Present.csv")

cleaned_df = df[["DATE OCC", "AREA", "AREA NAME", "Crm Cd", "Crm Cd Desc", ]]
cleaned_df = cleaned_df.dropna()

cleaned_df["DATE OCC"] = pd.to_datetime(cleaned_df["DATE OCC"])
cleaned_df["Year"] = cleaned_df["DATE OCC"].dt.year
cleaned_df["Month"] = cleaned_df["DATE OCC"].dt.month
cleaned_df = cleaned_df.drop(columns=["DATE OCC"])

cleaned_df.to_csv("cleaned_data.csv", index = False)