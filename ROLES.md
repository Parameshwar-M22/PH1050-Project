# Roles and Responsibilities

PH1050 Final Group Project, Problem 9: Atoms and the Periodic Table

The project has three parts. Parts 1 and 2 were each done by a pair, and Part 3 (the data-driven model) was done by Tanay and Parameshwar. Within each pair the work is split evenly.

## At a glance

| Member | Part | Main contribution |
|---|---|---|
| Adhirath | Part 1: Physics model | Lennard-Jones potential, minimum, numerical force and tolerance check |
| Yazhini | Part 1: Physics model | Convergence study, lowest-energy cluster search |
| Lohit | Part 2: Data and statistics | Data cleaning of the element table, noble-gas ratios |
| Vignesh | Part 2: Data and statistics | Statistics on the ratios, outlier test, interpretation |
| Tanay | Part 3: Data-driven model | scikit-learn 5-fold cross-validation, data cleaning and modelling, interpretation of results |
| Parameshwar | Part 3: Data-driven model | From-scratch implementations, KNN experiments, scikit-learn extrapolation |

## Part 1: Physics model (Adhirath and Yazhini)

Files: `minimum_energy.py`, `grid_minimizer.py`

### Adhirath
- Defined the Lennard-Jones potential and plotted $V(r)$ for $0.9 \le r \le 3$ (`lj_potential.png`).
- Derived the analytic minimum, $r^* = 2^{1/6}$ with $V(r^*) = -1$.
- Implemented the bisection root-finder (`find_minimum`) on $V'(r) = 0$ and checked it against the analytic value.
- Implemented the central-difference force (`F_numerical`) and the tolerance check against the analytic force (`F_analytic`).

### Yazhini
- Ran the $h$-convergence study (`convergence_study`) and fitted the truncation and round-off slopes (`fit_slope`, `fit_roundoff_slope`) for `force_convergence.png`.
- Wrote the cluster energy and analytic gradient (`energy`, `gradient`) in `grid_minimizer.py`.
- Implemented the random starts with overlap rejection (`random_start`) and the multi-start BFGS search (`best_cluster`).
- Produced the cluster results table and the energy-per-atom plot (`energy_per_atom.png`) for $N = 2$ to $7$.

## Part 2: Data and statistics (Lohit and Vignesh)

Files: `clean_elelments.py`, `part2_stats.py`, `ElementData.csv`

### Lohit
- Cleaned the element table: counted missing values per column and decided what to do with each (drop, fill or exclude).
- Counted the elements that have all required properties (72 of 118) and the 107 that have a boiling point.
- Computed $T_\text{melt}/T_\text{boil}$ for Ne, Ar, Kr and Xe and built the ratio table, excluding helium (no solid at 1 atm).

### Vignesh
- Computed the mean and spread of the ratios for Ne to Xe.
- Built the 95% prediction interval from Ar, Kr and Xe and tested neon against it (`part2_stats.py`).
- Produced the noble-gas plot (`noble_gas_ratio.png`).
- Interpreted the result: neon as a weak outlier, the caveats (only three reference points, neon chosen after looking) and the quantum-effects hypothesis.

## Part 3: Data-driven model (Tanay and Parameshwar)

### Tanay
- Scikit-learn 5-fold cross-validation (`Scikit_learn_based/5fcross_validation/`): baseline, linear regression, KNN and decision tree.
- Data cleaning and modelling for Part 3 (`data_model.py`): dropping elements without a boiling point, filling missing groups, imputing electronegativity inside the pipeline.
- Interpretation of the results: why the tree wins, why KNN and linear regression struggle in extrapolation.

### Parameshwar
- Built the algorithms from scratch without scikit-learn (`Without_scikit_learn/`): baseline, KNN, linear regression by gradient descent and the cross-validation splitter, for both 5-fold and extrapolation.
- Experimented with KNN variations: the $k = 1$ to $20$ sweeps for cross-validation and extrapolation and their plots.
- Scikit-learn extrapolation (`Scikit_learn_based/Extrapolation/`): training on periods 1 to 4 and testing on periods 5 to 6.

## Shared work

| Task | Who |
|---|---|
| Written report | `[confirm who]` |
| Repository README | `[confirm who]` |
| 12-minute talk | whole group, each member presents their own part |

---

*The split within Parts 1 and 2 is a proposed even division of that part's work. Adjust the bullets to match what each person actually did before submitting.*
