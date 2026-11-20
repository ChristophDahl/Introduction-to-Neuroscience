"""Demo 1 — train one artificial decision neuron on synthetic sensory activity.

Parallel to ANN_Demo1_SensoryDecision.m: same task, model, parameters,
learning rule, and plots. Python and MATLAB draw different random trials,
so individual numbers will not match exactly, even with seed 12.

Requires: NumPy, Matplotlib. Run: python ANN_Demo1_SensoryDecision.py
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(12)
sensory_noise = 0.6        # TRY: 0.2, 0.6, 1.2
n_train, n_test = 240, 120
learning_rate = 0.4
n_epochs = 400


def motion_trials(n, noise):
    """Synthetic left/right inputs; labels 0=left, 1=right.

    This is an illustrative sensory readout, NOT recorded MT neurons.
    """
    y = (rng.random(n) > 0.5).astype(float)
    X = noise * rng.standard_normal((n, 2))
    X[:, 0] += 0.9 * (1.0 - y)  # left-preferring input
    X[:, 1] += 0.9 * y          # right-preferring input
    return X, y


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


# Independent training and held-out test trials.
X_train, y_train = motion_trials(n_train, sensory_noise)
X_test, y_test = motion_trials(n_test, sensory_noise)

# Train a single sigmoid neuron by supervised gradient descent.
w = np.zeros(2)
b = 0.0
loss = np.zeros(n_epochs)
for epoch in range(n_epochs):
    p = sigmoid(X_train @ w + b)
    p_safe = np.clip(p, 1e-10, 1.0 - 1e-10)
    loss[epoch] = -np.mean(y_train*np.log(p_safe) + (1-y_train)*np.log(1-p_safe))

    error_signal = p - y_train
    grad_w = X_train.T @ error_signal / n_train
    grad_b = error_signal.mean()
    w -= learning_rate * grad_w
    b -= learning_rate * grad_b

p_test = sigmoid(X_test @ w + b)
pred_test = (p_test >= 0.5).astype(float)
accuracy = np.mean(pred_test == y_test)

print("DEMO 1 - NOISY SENSORY DECISIONS")
print(f"Noise level: {sensory_noise:.2f}")
print(f"Weights: left = {w[0]:+.3f}, right = {w[1]:+.3f}; bias = {b:+.3f}")
print(f"Test accuracy: {100*accuracy:.1f}% (chance expectation = 50%)")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(np.arange(1, n_epochs+1), loss, linewidth=2)
ax1.set(xlabel="Training epoch", ylabel="Cross-entropy loss",
        title="Training reduces prediction error")
ax1.grid(True)

left = y_test == 0
right = y_test == 1
ax2.scatter(X_test[left, 0], X_test[left, 1], label="Left trials", s=30)
ax2.scatter(X_test[right, 0], X_test[right, 1], label="Right trials", s=30)
xs = np.linspace(X_test[:, 0].min()-.2, X_test[:, 0].max()+.2, 150)
if abs(w[1]) > 1e-9:
    ys = -(w[0]*xs + b) / w[1]
    ax2.plot(xs, ys, "k--", linewidth=2, label="Decision boundary")
ax2.set(xlabel="Left-preferring input (synthetic)",
        ylabel="Right-preferring input (synthetic)",
        title=f"Unseen trials: {100*accuracy:.1f}% correct")
ax2.set_aspect("equal", adjustable="box")
ax2.legend()
ax2.grid(True)
fig.tight_layout()
plt.show()

# QUESTIONS
# 1. Does test accuracy decrease as sensory_noise increases?
# 2. Does changing noise change the input observations, the learning rule, or both?
# 3. What do the signs of the learned weights mean?
# 4. Why is performance evaluated on new trials?
# IMPORTANT: supervised gradient descent is NOT Hebbian LTP.
