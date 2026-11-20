"""Demo 2 — linear neuron versus one nonlinear hidden layer on XOR.

Parallel to ANN_Demo2_XOR.m: the same four examples, exact starting
weights, 4 tanh hidden units, sigmoid output, gradient descent, and plots.
The four examples are BOTH training and evaluation: this is NOT a test
of generalization to new data.

Requires: NumPy, Matplotlib. Run: python ANN_Demo2_XOR.py
"""
import numpy as np
import matplotlib.pyplot as plt

X = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
t = np.array([0., 1., 1., 0.])
n = len(t)
eta = 1.0
n_epochs = 7000


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


# First: one linear sigmoid neuron cannot separate XOR classes.
w_linear = np.zeros(2)
b_linear = 0.0
for epoch in range(n_epochs):
    p = sigmoid(X @ w_linear + b_linear)
    delta = (p - t)/n
    w_linear -= eta * (X.T @ delta)
    b_linear -= eta * delta.sum()
p_linear = sigmoid(X @ w_linear + b_linear)

# Next: 2 inputs -> 4 tanh hidden units -> 1 sigmoid output.
# Same fixed initial weights as the MATLAB version.
W1 = np.array([[0.6, -0.9, 0.5, 1.2],
               [0.8,  0.7, -1.1, -0.5]])
b1 = np.zeros(4)
W2 = np.array([0.3, -0.5, 0.8, -0.2])
b2 = 0.0
loss = np.zeros(n_epochs)

for epoch in range(n_epochs):
    H = np.tanh(X @ W1 + b1)
    p = sigmoid(H @ W2 + b2)
    p_safe = np.clip(p, 1e-10, 1-1e-10)
    loss[epoch] = -np.mean(t*np.log(p_safe) + (1-t)*np.log(1-p_safe))

    # Backpropagation in NumPy: propagate prediction error backward.
    delta_out = (p-t)/n
    grad_W2 = H.T @ delta_out
    grad_b2 = delta_out.sum()
    delta_hidden = np.outer(delta_out, W2)*(1-H**2)
    grad_W1 = X.T @ delta_hidden
    grad_b1 = delta_hidden.sum(axis=0)

    W1 -= eta*grad_W1
    b1 -= eta*grad_b1
    W2 -= eta*grad_W2
    b2 -= eta*grad_b2

H = np.tanh(X @ W1 + b1)
p_network = sigmoid(H @ W2 + b2)

print("DEMO 2 - XOR")
print("x1 x2 | target | one neuron | hidden layer")
for k in range(n):
    print(f" {int(X[k,0])}  {int(X[k,1])} |   {int(t[k])}    |   {p_linear[k]:.3f}    |    {p_network[k]:.3f}")
print(f"Linear training accuracy: {100*np.mean((p_linear>=0.5)==t):.0f}%")
print(f"Network training accuracy: {100*np.mean((p_network>=0.5)==t):.0f}%")
print("NOTE: these are the 4 training cases, not an independent test set.")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(np.arange(1, n_epochs+1), loss, linewidth=2)
ax1.set(xlabel="Training epoch", ylabel="Cross-entropy loss",
        title="Hidden-layer network learns XOR")
ax1.grid(True)

xx, yy = np.meshgrid(np.linspace(-.25, 1.25, 100),
                     np.linspace(-.25, 1.25, 100))
Q = np.column_stack((xx.ravel(), yy.ravel()))
HQ = np.tanh(Q @ W1 + b1)
pq = sigmoid(HQ @ W2 + b2).reshape(xx.shape)
ax2.contour(xx, yy, pq, levels=[0.5], linewidths=2)
ax2.scatter(X[t==0,0], X[t==0,1], marker="o", label="Target 0", s=100)
ax2.scatter(X[t==1,0], X[t==1,1], marker="s", label="Target 1", s=100)
ax2.set(xlabel="Input x1", ylabel="Input x2", title="Nonlinear decision boundary")
ax2.set(xlim=(-.3, 1.3), ylim=(-.3, 1.3))
ax2.set_aspect("equal", adjustable="box")
ax2.grid(True)
ax2.legend()
fig.tight_layout()
plt.show()

# QUESTIONS
# 1. Can one straight line separate the XOR classes?
# 2. Why does adding nonlinear hidden units change what the network can learn?
# 3. Is 100% accuracy on four training cases evidence of generalization? (No.)
# 4. Try reducing the hidden-unit count. Update W1, b1, W2 consistently.
# CAUTION: artificial hidden layers are not anatomical cortical layers.
