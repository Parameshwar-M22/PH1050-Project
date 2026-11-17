## How to Run

### 1. Run the Scikit-learn Implementation

**5-Fold Cross-Validation**

Navigate to the cross-validation folder:

```bash
cd Scikit_learn_based/5fcross_validation
```

Run:

```bash
python result.py
```

This evaluates the Baseline, Linear Regression, KNN and Decision Tree models using 5-fold cross-validation.

**Extrapolation**

Navigate to the Scikit-learn implementation folder:

```bash
cd ..
```

Run the extrapolation module:

```bash
python -m Extrapolation.result_ex
```

This evaluates the models by training on elements from Periods 1–4 and testing on elements from Periods 5–6.

### 2. Run the Implementation Without Scikit-learn

**5-Fold Cross-Validation**

Navigate to the folder:

```bash
cd ../../Without_scikit_learn/5fold_cross validation
```

Run:

```bash
python main.py
```

This runs the manually implemented Baseline, KNN and Linear Regression models and evaluates them using 5-fold cross-validation.

**Extrapolation**

Navigate to the extrapolation folder:

```bash
cd ../extrapolation
```

Run:

```bash
python main.py
```

This evaluates the manually implemented models using the extrapolation train-test split.

### 3. Run the Analysis Scripts

The project also contains scripts for analysing how model performance changes with different parameter values.

For example, inside `Without_scikit_learn/5fold_cross validation`, run:

```bash
python knn_analysis.py
```

The analysis scripts generate plots and display the corresponding MAE values.