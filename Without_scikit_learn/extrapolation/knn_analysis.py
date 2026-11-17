
import matplotlib.pyplot as plt

from extrapolation.data_split import split_data
from extrapolation.result import evaluate_model
from knn import knn_predict


# Get training and testing data
X_train, X_test, y_train, y_test = split_data()

# Try different values of k
k_values = range(1, 21)
mae_values = []

for k in k_values:

    mae = evaluate_model(
        knn_predict,
        X_train, X_test, y_train, y_test,
        scale=True,
        k=k
    )

    mae_values.append(mae)


# Plot the results
plt.figure(figsize=(8, 5))

plt.plot(k_values, mae_values, marker='o')

plt.xlabel("Number of Neighbours (k)")
plt.ylabel("Mean Absolute Error")
plt.title("KNN: k vs Extrapolation MAE")

plt.xticks(list(k_values))
plt.grid(True)

plt.savefig("extrapolation/knn_analysis.png", dpi=300)
plt.show()


# Print the results
for k, mae in zip(k_values, mae_values):
    print(f"k = {k}, MAE = {mae:.2f}")
