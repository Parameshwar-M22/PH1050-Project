"""
Lennard-Jones clusters, Part 1 (B2, B5)

Find the lowest-energy arrangement of N atoms in 3D, N = 2..7, by running
scipy.optimize.minimize from MANY random starting configurations and keeping
the best one. Then compare with the known answers and plot energy per atom.

Total energy:  E = sum over pairs i<j of 4 * (r_ij^-12 - r_ij^-6)
"""
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Known global-minimum energies (from the problem sheet).
# N=2 is exactly -1 (one pair at the minimum of V).
KNOWN = {2: -1.0, 3: -3.0, 4: -6.0, 5: -9.104, 6: -12.712, 7: -16.505}


# ----------------------------------------------------------------------
# Energy and its analytic gradient (vectorised, no Python loops)
# ----------------------------------------------------------------------
def _pair_data(x):
    """
    x is a flat array of length 3N (what scipy wants). Reshape to (N, 3) and
    build, for every pair at once:
        d[i, j] = pos[i] - pos[j]    (vector from j to i)
        r2[i, j] = |d[i, j]|^2
    The diagonal (i == j) has r2 = 0, which would give 1/0. We set it to 1
    as a dummy and zero its contribution afterwards with `mask`.
    """
    pos = x.reshape(-1, 3)
    d = pos[:, None, :] - pos[None, :, :]       # shape (N, N, 3)
    r2 = np.sum(d**2, axis=-1)                  # shape (N, N)
    mask = ~np.eye(len(pos), dtype=bool)        # True off the diagonal
    r2 = np.where(mask, r2, 1.0)
    return d, r2, mask


def energy(x):
    """
    Total LJ energy. Using r2 = r^2:  r^-12 = r2^-6,  r^-6 = r2^-3.
    The (N, N) matrix counts every pair twice (i,j and j,i), hence the 0.5.
    (Your version did the same sum with a double Python loop, which is
    slow and hard for scipy to differentiate.)
    """
    _, r2, mask = _pair_data(x)
    pair = 4 * (r2**-6 - r2**-3)
    return 0.5 * np.sum(pair * mask)


def gradient(x):
    """
    Analytic gradient dE/dx. For atom i:

        dE/d(pos_i) = sum_j  V'(r_ij) * (pos_i - pos_j) / r_ij
        V'(r)       = -48 r^-13 + 24 r^-7
        => V'(r)/r  = -48 r^-14 + 24 r^-8 = -48 r2^-7 + 24 r2^-4

    Giving scipy the exact gradient makes BFGS faster and more accurate
    than letting it estimate the gradient by finite differences.
    """
    d, r2, mask = _pair_data(x)
    coef = (-48 * r2**-7 + 24 * r2**-4) * mask   # shape (N, N)
    grad = np.sum(coef[:, :, None] * d, axis=1)  # sum over j -> shape (N, 3)
    return grad.ravel()


# ----------------------------------------------------------------------
# Random starts and the search
# ----------------------------------------------------------------------
def random_start(N, rng, min_sep=0.7):
    """
    Random positions in a cube about 2.4 wide: uniform(-1.2, 1.2) for each of
    the 3N coordinates. This is the box suggested in the instructor's hints
    (your old box=5 was far too empty: atoms started so far apart that the
    forces were tiny and the minimiser could wander).

    The hint warns that atoms starting almost on top of each other give huge
    energies. We go one step further and REJECT starts where two atoms are
    closer than min_sep, so BFGS never sees an inf/nan. (min_sep = 0.7 is my
    choice, not the hint's; set it to 0 to follow the hint literally.)
    """
    while True:
        pos = rng.uniform(-1.2, 1.2, size=(N, 3))
        d = pos[:, None, :] - pos[None, :, :]
        r = np.sqrt(np.sum(d**2, axis=-1))
        if r[np.triu_indices(N, 1)].min() > min_sep:
            return pos.ravel()


def best_cluster(N, n_starts=60, seed=0):
    """
    Run BFGS from n_starts random configurations (60, as in the instructor's
    hints) and keep the lowest energy.
    Why many starts: for N >= 6 the energy surface has several local minima,
    and a single start often lands in one that is not the global minimum.

    Returns (best energy, best positions, list of all converged energies).
    """
    rng = np.random.default_rng(seed)   # seeded -> reproducible results
    best, energies = None, []
    for _ in range(n_starts):
        res = minimize(energy, random_start(N, rng), jac=gradient,
                       method="BFGS", options={"gtol": 1e-6})
        # Accept a run if the gradient is genuinely ~0 (we are at a minimum),
        # NOT if res.success is True. With a very strict gtol (I first used
        # 1e-8), BFGS often stops with "precision loss" and success=False even
        # though it is essentially at a minimum; throwing those runs away
        # silently biased the search and made N=6 miss -12.712.
        if np.linalg.norm(gradient(res.x)) > 1e-4:
            continue
        energies.append(res.fun)
        if best is None or res.fun < best.fun:
            best = res
    return best.fun, best.x.reshape(-1, 3), np.array(energies)


# ----------------------------------------------------------------------
# Run everything
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Sanity check: analytic gradient vs finite differences on a random cluster.
    from scipy.optimize import check_grad
    rng = np.random.default_rng(1)
    x_test = random_start(5, rng)
    # check_grad returns the norm of (analytic - finite-difference) gradient.
    # It should be tiny compared with the gradient itself.
    print("gradient check: difference =", check_grad(energy, gradient, x_test),
          " vs |gradient| =", np.linalg.norm(gradient(x_test)))

    Ns = np.arange(2, 8)                       # N = 2..7 (old code stopped at 6)
    E = np.zeros(len(Ns))
    print(f"\n{'N':>2} {'E found':>10} {'E known':>10} {'diff':>9} "
          f"{'hit rate':>9}")
    for i, N in enumerate(Ns):
        E[i], _, all_E = best_cluster(N)
        # hit rate = fraction of starts that reached the global minimum;
        # it falls as N grows, which is why one start is not enough.
        hit = np.mean(np.abs(all_E - E[i]) < 1e-4)
        print(f"{N:>2} {E[i]:>10.4f} {KNOWN[N]:>10.4f} "
              f"{E[i] - KNOWN[N]:>9.1e} {hit:>9.0%}")
        # tolerance 2e-3 because the sheet quotes 3 decimals
        assert abs(E[i] - KNOWN[int(N)]) < 2e-3, f"N={N} does not match"

    plt.figure(figsize=(6, 4.5))
    plt.plot(Ns, E / Ns, "-o")
    plt.title("Lennard-Jones clusters: energy per atom vs N")
    plt.xlabel("Number of atoms N")
    plt.ylabel("Energy per atom (units of epsilon)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("energy_per_atom.png", dpi=200)
    plt.show()
