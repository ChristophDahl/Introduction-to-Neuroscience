"""Systematic Hebbian parameter sweep for an Introduction to Neuroscience class.

Run:
    python hebbian_parameter_sweep.py

Requires: numpy, matplotlib (no pandas required).

Learning rule: w(t+1) = w(t) + eta * x(t) * y(t)

Full factorial: 3 learning rates x 2 firing rates x 5 co-activation
levels x 3 trial counts x 2 starting weights = 180 conditions.

The design holds the overall firing rates of the two neurons equal
within each condition, while varying their overlap (co-activation).

Co-activation c = P(y=1 | x=1), with exactly p*N spikes in each neuron.
The expected coincidence for statistically independent activity is c=p.
The special case c=0 means NO coincidences (not independence).

For these constructed binary patterns: w_final = w0 + eta*N*p*c.
"""

import csv
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Reproducible randomized ordering of binary activities.
rng = np.random.default_rng(7)

# Parameters to change for class demonstrations.
eta_values = [0.005, 0.020, 0.050]
p_values = [0.20, 0.40]
c_values = [0, 0.25, 0.50, 0.75, 1.00]
N_values = [40, 80, 160]
w0_values = [0.20, 0.50]

outdir = Path("hebbian_outputs")
outdir.mkdir(exist_ok=True)


def make_activity(N: int, p: float, c: float, generator: np.random.Generator):
    """Binary pre/post activity with controlled marginal rates and overlap.

    Each neuron fires nx=p*N times; exactly no=c*nx are shared trials.
    Requires integer spike counts. p <= 0.5 permits c=0 at equal rates.
    """
    if not (0 <= p <= 0.5 and 0 <= c <= 1):
        raise ValueError("Require 0 <= p <= 0.5 and 0 <= c <= 1")

    nx = round(N * p)
    no = round(nx * c)
    if not np.isclose(nx, N * p) or not np.isclose(no, nx * c):
        raise ValueError("Choose N, p, c so all spike counts are integers")

    x = np.zeros(N, dtype=np.int8)
    y = np.zeros(N, dtype=np.int8)
    active = generator.choice(N, size=nx, replace=False)
    x[active] = 1

    if no:
        y[generator.choice(active, size=no, replace=False)] = 1
    remaining_y = nx - no
    if remaining_y:
        inactive = np.flatnonzero(x == 0)
        y[generator.choice(inactive, size=remaining_y, replace=False)] = 1

    assert x.sum() == nx and y.sum() == nx and (x * y).sum() == no
    return x, y


# 1. Systematic factorial sweep and CSV export.
fieldnames = ["eta", "p", "c", "N", "w0", "weight_change", "final_weight",
              "observed_c", "predicted_change"]
results = []

for eta, p, c, N, w0 in product(eta_values, p_values, c_values, N_values, w0_values):
    x, y = make_activity(N, p, c, rng)

    weight = np.empty(N + 1)
    weight[0] = w0
    for t in range(N):
        weight[t + 1] = weight[t] + eta * x[t] * y[t]

    change = float(weight[-1] - w0)
    observed_c = float(np.dot(x, y) / x.sum())
    predicted_change = float(eta * N * p * c)
    assert np.isclose(change, predicted_change, atol=1e-10)

    results.append({
        "eta": eta, "p": p, "c": c, "N": N, "w0": w0,
        "weight_change": change, "final_weight": float(weight[-1]),
        "observed_c": observed_c, "predicted_change": predicted_change,
    })

csv_path = outdir / "hebbian_factorial_results.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

assert len(results) == 180
print(f"Completed {len(results)} conditions.")
print(f"Saved results to: {csv_path.resolve()}")

# 2. Figure 1: activities, learning trajectories, and parameter interactions.
fig1, ax = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")

eta, p, N, w0 = 0.02, 0.40, 80, 0.20
for c in [0, 0.50, 1]:
    x, y = make_activity(N, p, c, rng)
    weight = w0 + eta * np.concatenate(([0], np.cumsum(x * y)))
    ax[0, 0].step(np.arange(N + 1), weight, where="post", linewidth=2, label=f"c = {c:g}")
ax[0, 0].set(title="A. Change co-activation; hold firing rates constant",
             xlabel="Trial", ylabel="Synaptic weight")
ax[0, 0].legend(loc="upper left")
ax[0, 0].grid(alpha=0.25)

x, y = make_activity(40, 0.40, 0.50, rng)
ax[0, 1].vlines(np.flatnonzero(x) + 1, 0.8, 1.2, linewidth=1.8, label="Pre (x)")
ax[0, 1].vlines(np.flatnonzero(y) + 1, 1.8, 2.2, linewidth=1.8, label="Post (y)")
ax[0, 1].set(xlim=(0, 41), ylim=(0.5, 2.5), xlabel="Trial",
             title="B. Same firing rates; some coincidences")
ax[0, 1].set_yticks([1, 2], ["Pre (x)", "Post (y)"])
ax[0, 1].grid(alpha=0.20)

heat = np.outer(eta_values, c_values) * (80 * 0.40)
im = ax[1, 0].imshow(heat, origin="lower", aspect="auto")
ax[1, 0].set_xticks(range(len(c_values)), [str(v) for v in c_values])
ax[1, 0].set_yticks(range(len(eta_values)), [str(v) for v in eta_values])
ax[1, 0].set(xlabel="Co-activation (c)", ylabel="Learning rate (eta)",
             title="C. Weight change: learning rate × co-activation")
fig1.colorbar(im, ax=ax[1, 0], label="Weight change")

for p in p_values:
    final_weights = 0.20 + 0.02 * np.array(N_values) * p * 0.75
    ax[1, 1].plot(N_values, final_weights, "-o", linewidth=2, label=f"p = {p:g}")
ax[1, 1].set(title="D. Practice × activity rate (c = 0.75)",
             xlabel="Number of trials (N)", ylabel="Final synaptic weight")
ax[1, 1].legend(loc="upper left")
ax[1, 1].grid(alpha=0.25)

fig1.savefig(outdir / "figure1_hebbian_parameter_effects.png", dpi=200, bbox_inches="tight")

# 3. Figure 2: effect of the initial weight.
fig2, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
w0_arr = np.array(w0_values)
for c in c_values:
    gain = 0.02 * 80 * 0.40 * c
    axes[0].plot(w0_arr, w0_arr + gain, "-o", label=f"c = {c:g}")
    axes[1].plot(w0_arr, np.full_like(w0_arr, gain), "-o", label=f"c = {c:g}")
axes[0].set(xlabel="Initial weight (w0)", ylabel="Final weight",
            title="E. Initial weight shifts the outcome")
axes[1].set(xlabel="Initial weight (w0)", ylabel="Weight change (delta w)",
            title="F. Initial weight does not change learning")
axes[0].legend(loc="upper left")
for axis in axes:
    axis.grid(alpha=0.25)
fig2.savefig(outdir / "figure2_initial_weight.png", dpi=200, bbox_inches="tight")

print("Prediction: w_final = w0 + eta*N*p*c")
print(f"Example: w0=.2, eta=.02, N=80, p=.4, c=.5 -> w_final={0.2+0.02*80*0.4*0.5:.2f}")
print("Independence corresponds to c approximately p (not c = 0).")

plt.show()

# Excluded: LTD, STDP, weight saturation, intrinsic and homeostatic plasticity.
