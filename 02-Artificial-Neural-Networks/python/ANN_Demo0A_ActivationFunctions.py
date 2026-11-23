"""ANN Demo 0A: compare six activation functions (NumPy + Matplotlib).

Run: python ANN_Demo0A_ActivationFunctions.py
These curves act on the neuron's weighted sum z. Values are illustrative.
"""
import numpy as np
import matplotlib.pyplot as plt

z = np.linspace(-5, 5, 501)
alpha = 0.10                     # Leaky ReLU negative slope: TRY 0.01 / 0.3

functions = {
    "Step": ((z >= 0).astype(float), (-0.1, 1.1)),
    "Sigmoid": (1.0 / (1.0 + np.exp(-z)), (-0.1, 1.1)),
    "Tanh": (np.tanh(z), (-1.2, 1.2)),
    "ReLU": (np.maximum(0, z), (-0.5, 5.2)),
    "Leaky ReLU": (np.maximum(alpha*z, z), (-1.2, 5.2)),
    "Linear": (z, (-5.2, 5.2)),
}

fig, axes = plt.subplots(2, 3, figsize=(11, 6.5))
for ax, (name, (values, y_range)) in zip(axes.flat, functions.items()):
    ax.plot(z, values, color="teal", lw=2)
    ax.axvline(0, color="0.6", lw=1, ls=":")
    ax.axhline(0, color="0.6", lw=1, ls=":")
    ax.set(xlim=(-5, 5), ylim=y_range, title=name,
           xlabel="Weighted input z", ylabel="Output f(z)")
    ax.grid(alpha=0.3)
fig.tight_layout()
print("For z=2: sigmoid=%.3f; tanh=%.3f; ReLU=2; Leaky ReLU=2" %
      (1/(1+np.exp(-2)), np.tanh(2)))
plt.show()

# TRY:
# 1. Change alpha. How does the negative side of Leaky ReLU change?
# 2. Which functions permit negative or unbounded outputs?
# 3. Is the ReLU output a probability? Why not?
# 4. A sigmoid output in [0,1] is NOT automatically calibrated.
