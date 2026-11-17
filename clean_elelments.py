import pandas as pd

# load the raw table (118 elements, 30 columns)
df = pd.read_csv("elements.csv")
print(df.shape)

# the 7 properties used in Problem 9
props = ["AtomicNumber", "AtomicMass", "Period", "Group",
         "Electronegativity", "MeltingPoint", "BoilingPoint"]

# keep only these columns, so unused empty columns don't remove elements
data = df[["EnglishName", "Symbol"] + props].copy()

# make all values numeric; text entries become NaN (missing)
for p in props:
    data[p] = pd.to_numeric(data[p], errors="coerce")

# count missing values in each property
print(data[props].isna().sum())

# keep only elements that have all 7 properties (no filling in values)
clean = data.dropna(subset=props)

print("total elements:", len(data))
print("elements with all properties:", len(clean))
print("dropped:", len(data) - len(clean))

# noble gases are checked in the full table, since He, Ne, Ar lack electronegativity
# and are missing from clean; helium has no melting point
noble = data[data["Symbol"].isin(["He", "Ne", "Ar", "Kr", "Xe"])]
print(noble[["Symbol", "MeltingPoint", "BoilingPoint"]])

# save the cleaned table for Part 3
clean.to_csv("elements_clean.csv", index=False)
