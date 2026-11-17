#import pandas
import pandas as pd
#convert the csv file into a dataframe
from pathlib import Path
csv_path = Path(__file__).parent / "ElementData.csv"
df = pd.read_csv(csv_path)
#Checking for missing values.
print("--CHECKING FOR MISSING VALUES--")
#print(df.isnull().sum())          
#Checking for missing values specially in the BP to remove them
print("No.of elements with missing BP values:" ,df["BoilingPoint"].isnull().sum())
df = df.dropna(subset=["BoilingPoint"])
 #Adding group number 3 to all Lanthanides and Actinides
df["Group"] = df["Group"].fillna(3)      
#Creating the input and output label pairs:
X = df[["AtomicNumber", "AtomicMass", "Period", "Group", "Electronegativity"]]
y = df[["BoilingPoint"]]


