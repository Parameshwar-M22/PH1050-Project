### README

**PART 3** 
**Explanation**:
# Predicting Boiling Points of Elements 
In this part of the project we try to predict the boiling point of an element from a few of its properties: atomic number, atomic mass, period, group and electronegativity.
The idea is that a boiling point depends on how strongly atoms attract each other, and these properties are linked to that. We compared four models: a baseline, linear regression, k-nearest neighbours (KNN) and a decision tree. Each is written with scikit-learn and from scratch in NumPy, except the tree, which is scikit-learn only. They are tested with 5-fold cross-validation. Secondly we tried an extrapolation test by training on periods 1–4 and testing on periods 5–6.
Finally we tried to analyze why we got the particular mae values.

The decision tree did best in both tests, but even its error is a few hundred kelvin, so these properties alone are not enough to predict boiling points accurately.
**#Required dependencies**
1.pandas
2.numpy
3.scikit_learn
4.Python virtual environment

# How to Run


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
