"""Hebbian learning: paired vs. unpaired activity.

Introduction to Neuroscience — Neural Plasticity

Run:
    python hebbian_learning_demo.py

Requires: numpy, matplotlib

Learning rule: w(t+1) = w(t) + eta * x(t) * y(t)
Here x and y are binary activities (0 = inactive; 1 = active).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Parameters: edit these values and rerun.
N = 40                 # Number of trials
eta = 0.02             # Learning rate
w0 = 0.20              # Initial synaptic strength

# Neuron A fires on every second trial. Both B conditions fire equally often.
trials = np.arange(1, N + 1)
x = (trials % 2 == 0).astype(int)
y_paired = x.copy()        # Always active with neuron A
y_unpaired = 1 - x         # Never active with neuron A

# Record the weight at baseline and after each learning trial.
w_paired = np.empty(N + 1)
w_unpaired = np.empty(N + 1)
w_paired[0] = w0
w_unpaired[0] = w0

for t in range(N):
    w_paired[t + 1] = w_paired[t] + eta * x[t] * y_paired[t]
    w_unpaired[t + 1] = w_unpaired[t] + eta * x[t] * y_unpaired[t]

# Figure: activity and resulting weight changes.
fig, axes = plt.subplots(2, 1, figsize=(10, 7), layout="constrained")

activity = np.vstack([x, y_paired, y_unpaired])
axes[0].imshow(activity, cmap="Greys", interpolation="nearest", aspect="auto", vmin=0, vmax=1)
axes[0].set_yticks([0, 1, 2], ["Neuron A", "Neuron B (paired)", "Neuron B (unpaired)"])
axes[0].set_xticks(np.arange(0, N, 5), np.arange(1, N + 1, 5))
axes[0].set_xlabel("Trial")
axes[0].set_title("Neuronal activity (dark = active)")

axes[1].plot(np.arange(N + 1), w_paired, label="Paired activity", linewidth=2)
axes[1].plot(np.arange(N + 1), w_unpaired, "--", label="Unpaired activity", linewidth=2)
axes[1].set_xlabel("Trial")
axes[1].set_ylabel("Synaptic weight (w)")
axes[1].set_title(r"Hebbian learning: $\Delta w = \eta xy$")
axes[1].legend(loc="upper left")
axes[1].grid(alpha=0.25)

outdir = Path("hebbian_outputs")
outdir.mkdir(exist_ok=True)
fig.savefig(outdir / "figure0_hebbian_demo.png", dpi=200, bbox_inches="tight")

print(f"Final weight, paired:   {w_paired[-1]:.2f}")
print(f"Final weight, unpaired: {w_unpaired[-1]:.2f}")
print(f"Saved figure to: {outdir / 'figure0_hebbian_demo.png'}")
plt.show()

# Deliberately excluded: weight saturation, LTD, STDP, homeostasis.
