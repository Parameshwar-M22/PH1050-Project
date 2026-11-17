import matplotlib.pyplot as plt
from data_model import X, y
from result import evaluate_model
from knn import knn_predict
k_values = range(1, 21)
mae_values = []

for k in k_values:
    mae = evaluate_model(
        knn_predict,
        X, y,
        scale=True,
        k=k
    )

    mae_values.append(mae)

plt.figure(figsize=(8, 5))
plt.plot(k_values, mae_values, marker='o')

plt.xlabel("Number of Neighbours (k)")
plt.ylabel("Average 5-Fold CV MAE")
plt.title("KNN: k vs Cross-Validation MAE")

plt.xticks(list(k_values))
plt.grid(True)
plt.tight_layout()

plt.savefig("knn_cv_analysis.png", dpi=300)
plt.show()

for k, mae in zip(k_values, mae_values):
    print(f"k = {k}, MAE = {mae:.2f}")