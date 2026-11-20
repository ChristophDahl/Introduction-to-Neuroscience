"""Demo 0 — one artificial neuron (parallel to ANN_Demo0_OneNeuron.m).

Requires only Python and NumPy. These input values are illustrative, not neural recordings.
Run: python ANN_Demo0_OneNeuron.py
"""
import numpy as np

# Inputs: synthetic activity of left- and right-preferring populations.
x = np.array([0.8, 0.2])
w = np.array([-2.0, 2.0])  # weights chosen by hand, not learned here
b = 0.0                    # decision bias

z = float(x @ w + b)
p_right = 1.0 / (1.0 + np.exp(-z))

print(f"Weighted input z = {z:.2f}")
print(f"P(right) = {p_right:.3f}")
print("Decision:", "RIGHT" if p_right >= 0.5 else "LEFT")

# EXERCISES
# 1. Change x to [0.2, 0.8]. What happens?
# 2. Set both weights to zero. What does the neuron predict?
# 3. Change b to +2. How does bias affect the decision?
# 4. Why is this NOT yet a learning demonstration?
