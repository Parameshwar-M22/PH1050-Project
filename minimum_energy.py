"""
Lennard-Jones potential, Part 1 (B2, B3, B4)

    V(r) = 4 * (r^-12 - r^-6)        (epsilon = sigma = 1)

What this script does
  1. Finds the minimum of V by BISECTION ON V'(r) = 0  (not on V(r) = -1).
  2. Checks that r* = 2^(1/6) and V(r*) = -1.
  3. Plots V(r) for 0.9 <= r <= 3 with the minimum marked (instructor's
     "start here" hint).
  4. Computes the force F = -dV/dr by central differences and compares it
     with the analytic force. Plots error vs step h on a log-log plot and
     measures the slope in BOTH regimes (instructor's hint): ~ +2 where
     truncation error dominates, ~ -1 where round-off dominates.

Layout: everything is a function, and the "run it" part sits under
`if __name__ == "__main__":`. That way you can `import` these functions in
your test file without a plot popping up.
"""
import numpy as np
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# The physics: defined ONCE (your old file defined V twice as lambdas)
# ----------------------------------------------------------------------
def V(r):
    """Lennard-Jones potential with epsilon = sigma = 1."""
    return 4 * (r**-12 - r**-6)


def F_analytic(r):
    """Analytic force F = -dV/dr = 48 r^-13 - 24 r^-7."""
    return 48 * r**-13 - 24 * r**-7


# ----------------------------------------------------------------------
# Numerical derivative / force
# ----------------------------------------------------------------------
def central_diff(f, x, h):
    """Central difference (f(x+h) - f(x-h)) / 2h.  Truncation error ~ h^2."""
    return (f(x + h) - f(x - h)) / (2 * h)


def F_numerical(r, h):
    """Force from V by central differences:  F = -dV/dr."""
    return -central_diff(V, r, h)


# ----------------------------------------------------------------------
# Root finding: bisection on V'(r) = 0
# ----------------------------------------------------------------------
def find_minimum(dV, a, b, tol=1e-12):
    """
    Bisection on the derivative dV, assuming dV(a) < 0 < dV(b)
    (the potential is falling at a and rising at b, so a minimum lies between).

    Why V'(r)=0 and not V(r)=-1?  At the minimum V just TOUCHES -1 (it never
    crosses it), so V(r)+1 has no sign change and bisection cannot work.
    V'(r) goes from negative to positive across the minimum: a clean sign
    change, which is exactly what bisection needs.

    Changes vs. your version:
      * dV is evaluated ONCE per step (before, twice).
      * `if / else` instead of two separate `if`s, so exactly one endpoint
        always moves and the loop cannot stall if dV(mid) == 0.
      * It takes the derivative function as an argument, so the same routine
        works with the analytic or the numerical derivative.
    """
    lo, hi = a, b
    assert dV(lo) < 0 < dV(hi), "interval does not bracket a minimum"
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if dV(mid) < 0:      # still falling -> minimum is to the right
            lo = mid
        else:                # rising (or flat) -> minimum is to the left
            hi = mid
    return 0.5 * (lo + hi)


# ----------------------------------------------------------------------
# Convergence study of the central-difference force
# ----------------------------------------------------------------------
def convergence_study(r0=1.5, k_values=range(2, 27)):
    """
    Error |F_numerical - F_analytic| at r0 for h = 2^-k.

    r0 = 1.5 is not the minimum. At r* the force is exactly 0, which is a
    poor place to look at errors. Any r0 away from r* works.

    Returns h, error arrays.
    """
    h = 2.0 ** -np.array(list(k_values))
    err = np.abs(F_numerical(r0, h) - F_analytic(r0))
    return h, err


def fit_slope(h, err, h_min=1e-4):
    """
    Slope of log(err) vs log(h), fitted ONLY where truncation error dominates.

    Error model:  err ~ A h^2   (truncation, wins at large h)
                      + B / h   (roundoff, wins at tiny h)
    Roundoff comes from subtracting two nearly equal numbers, V(r+h) - V(r-h),
    then dividing by a tiny 2h. So for very small h the error goes UP again.
    Fitting the whole range would give a meaningless slope; we fit h > h_min.
    h_min = 1e-4 is a reasonable cut-off; check it against your plot.
    """
    use = h > h_min
    slope, intercept = np.polyfit(np.log(h[use]), np.log(err[use]), 1)
    return slope, use


def fit_roundoff_slope(h, err, h_max=1e-7):
    """
    Slope in the ROUND-OFF regime (very small h), where err ~ B/h, so the
    log-log slope should be about -1. Hint from the instructor: show both
    regimes. h < 1e-7 is where roundoff clearly beats truncation at r0 = 1.5
    (the crossover is near h ~ 1e-5 to 1e-6; we stay well past it).
    The data are noisy here (roundoff is random-ish), so expect a rough -1.
    """
    use = h < h_max
    slope, _ = np.polyfit(np.log(h[use]), np.log(err[use]), 1)
    return slope, use


# ----------------------------------------------------------------------
# Run everything
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # ---- 0. Look at the potential first (instructor's "start here") --
    r_plot = np.linspace(0.9, 3.0, 400)
    plt.figure(figsize=(6, 4.5))
    plt.plot(r_plot, V(r_plot), label="V(r)")
    plt.axhline(0, color="grey", lw=0.8)
    plt.plot(2 ** (1 / 6), -1, "ro", label="minimum: r = 2^(1/6), V = -1")
    plt.ylim(-1.5, 3)          # V(0.9) ~ 6 would squash the interesting part
    plt.xlabel("r (units of sigma)"); plt.ylabel("V (units of epsilon)")
    plt.title("Lennard-Jones potential"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("lj_potential.png", dpi=200)

    # ---- 1. Locate the minimum (B2) ---------------------------------
    dV = lambda r: -F_analytic(r)          # V'(r) = -F(r)
    r_star = find_minimum(dV, 1.0, 2.0)    # V' < 0 at 1.0, V' > 0 at 2.0
    print(f"r*        = {r_star:.12f}")
    print(f"2^(1/6)   = {2 ** (1 / 6):.12f}")
    print(f"V(r*)     = {V(r_star):.12f}   (expected -1)")
    assert abs(r_star - 2 ** (1 / 6)) < 1e-9
    assert abs(V(r_star) + 1) < 1e-12

    # Same thing, but using the NUMERICAL derivative inside the bisection.
    # Only accurate to ~1e-10 because of the derivative's own error.
    r_star_num = find_minimum(lambda r: central_diff(V, r, 1e-6), 1.0, 2.0, tol=1e-9)
    print(f"r* (bisection with numerical V') = {r_star_num:.9f}")

    # ---- 2. Force check with a stated tolerance (B3) ----------------
    TOL = 1e-5   # truncation error ~ (h^2/6)|V'''|, and V''' is large at small r,
    #              so with h = 1e-4 a 1e-5 relative tolerance is realistic
    rs = np.linspace(1.0, 3.0, 50)
    worst = np.max(np.abs(F_numerical(rs, 1e-4) - F_analytic(rs)) / (1 + np.abs(F_analytic(rs))))
    print(f"max relative-ish force error (h=1e-4, r in [1,3]) = {worst:.2e}")
    print("B3 tolerance", TOL, "->", "PASS" if worst < TOL else "FAIL")

    # ---- 3. Convergence study (B4) ----------------------------------
    h, err = convergence_study()
    slope, used = fit_slope(h, err)
    print(f"truncation regime (h > 1e-4): slope = {slope:.3f}   (expected +2)")
    slope_r, used_r = fit_roundoff_slope(h, err)
    print(f"round-off regime  (h < 1e-7): slope = {slope_r:.3f}   (expected about -1)")
    # h at which the two error sources are equal (where the curve bottoms out)
    print(f"smallest error {err.min():.1e} at h = {h[np.argmin(err)]:.1e}")

    plt.figure(figsize=(6, 4.5))
    plt.loglog(h, err, "o-", label="|F_numerical - F_analytic|")
    # reference line with slope exactly 2, anchored on the first fitted point
    # so it sits on the data instead of floating parallel to it
    i0 = np.argmax(used)
    plt.loglog(h, err[i0] * (h / h[i0]) ** 2, "--", label="slope 2 reference")
    j0 = np.argmax(used_r)
    plt.loglog(h, err[j0] * (h / h[j0]) ** -1, "--", label="slope -1 reference (round-off)")
    plt.axvline(1e-4, color="grey", ls=":", label="truncation fit: h > 1e-4")
    plt.axvline(1e-7, color="grey", ls=":", label="round-off fit: h < 1e-7")
    plt.xlabel("step h")
    plt.ylabel("absolute error in force at r = 1.5")
    plt.title(f"Central-difference error: slopes {slope:.2f} and {slope_r:.2f}")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("force_convergence.png", dpi=200)
    plt.show()
