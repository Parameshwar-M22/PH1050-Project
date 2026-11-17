# ---------DATA CLEANING -------------
import pandas as pd
#convert the csv file into a dataframe
df = pd.read_csv("ElementData.csv")
#Checking for missing values.
#print(df.isnull().sum())          
#Checking for missing values specially in the BP to remove them
print(df["BoilingPoint"].isnull().sum())
df = df.dropna(subset=["BoilingPoint"])
 #Adding group number 3 to all Lanthanides and Actinides
df["Group"] = df["Group"].fillna(3) 
#Adding EN value of 1.25 (which was the median of all the existing values)     
df["Electronegativity"] = df["Electronegativity"].fillna(1.25)   
#Creating the input and output label pairs:
X = df[["AtomicNumber", "AtomicMass", "Period", "Group", "Electronegativity"]].to_numpy(dtype=float)
y = df["BoilingPoint"].to_numpy(dtype=float)