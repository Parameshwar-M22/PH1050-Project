#import the cleaned dataset so that it could be split
from .data_model import df
#Spilt the dataset into training and testing 
#Training:Elements in period<=4 and Testing: Elements in period 5 and 6
train_df = df[df["Period"] <= 4]
test_df = df[df["Period"].isin([5, 6])]
#Creating the input-output pair labels.
X_train = train_df[["AtomicNumber", "AtomicMass", "Period", "Group", "Electronegativity"]]
y_train = train_df[["BoilingPoint"]]
X_test = test_df[["AtomicNumber", "AtomicMass", "Period", "Group", "Electronegativity"]]
y_test = test_df[["BoilingPoint"]]
