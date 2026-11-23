"""Demo 3: task-context-dependent decisions. Synthetic stimuli; not recorded neurons.

Three inputs: motion evidence, colour evidence, task cue (1=motion, 0=colour).
Target: sign of the *cued* evidence. Compare a linear sigmoid unit with an
8-tanh-unit hidden-layer network. No deep-learning framework required.

Run: python ANN_Demo3_ContextDependentDecisions.py
Requires: numpy, matplotlib
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(22)
n_train, n_test = 800, 400
n_epochs = 1800
learning_rate = 0.4
n_hidden = 10
noise_sd = 0.08                # TRY: 0.0, 0.08, 0.25


def make_trials(n):
    motion = rng.choice([-1., 1.], n) * rng.uniform(.35, 1.0, n)
    colour = rng.choice([-1., 1.], n) * rng.uniform(.35, 1.0, n)
    context = rng.integers(0, 2, n).astype(float)   # 1 motion; 0 colour
    target = (np.where(context == 1, motion, colour) > 0).astype(float)
    X = np.column_stack((motion + rng.normal(0, noise_sd, n),
                         colour + rng.normal(0, noise_sd, n), context))
    conflict = (motion > 0) != (colour > 0)
    return X, target, conflict


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


X_train, y_train, _ = make_trials(n_train)
X_test, y_test, conflict = make_trials(n_test)

# Model 1: one sigmoid neuron, trained with the same labeled trials.
w = np.zeros(3)
b = 0.
for epoch in range(n_epochs):
    p = sigmoid(X_train @ w + b)
    err = p - y_train
    w -= learning_rate * (X_train.T @ err) / n_train
    b -= learning_rate * err.mean()

# Model 2: a hidden layer (tanh) followed by a sigmoid output.
W1 = rng.normal(0, 0.5, (3, n_hidden))
b1 = np.zeros(n_hidden)
W2 = rng.normal(0, 0.5, (n_hidden, 1))
b2 = 0.
train_loss = []
for epoch in range(n_epochs):
    H = np.tanh(X_train @ W1 + b1)
    p = sigmoid(H @ W2 + b2).ravel()
    if epoch % 20 == 0:
        q = np.clip(p, 1e-9, 1 - 1e-9)
        train_loss.append(-np.mean(y_train * np.log(q) + (1-y_train)*np.log(1-q)))
    delta2 = ((p-y_train)/n_train)[:, None]
    grad_W2 = H.T @ delta2
    grad_b2 = delta2.sum()
    delta1 = (delta2 @ W2.T) * (1-H**2)
    grad_W1 = X_train.T @ delta1
    grad_b1 = delta1.sum(axis=0)
    W2 -= learning_rate * grad_W2
    b2 -= learning_rate * grad_b2
    W1 -= learning_rate * grad_W1
    b1 -= learning_rate * grad_b1


def predict_hidden(X):
    return sigmoid(np.tanh(X @ W1 + b1) @ W2 + b2).ravel()


def accuracy(p, y):
    return 100 * np.mean((p >= .5) == y)


p_linear = sigmoid(X_test @ w + b)
p_hidden = predict_hidden(X_test)
X_cue_shuffled = X_test.copy()
X_cue_shuffled[:, 2] = rng.permutation(X_test[:, 2])  # break cue/target correspondence
p_no_cue = predict_hidden(X_cue_shuffled)

print("CONTEXT-DEPENDENT DECISIONS (SIMULATED DATA)")
print("Input 1: motion; input 2: colour; input 3: task cue")
print(f"Linear unit:        {accuracy(p_linear,y_test):.1f}% test accuracy")
print(f"Hidden-layer ANN:   {accuracy(p_hidden,y_test):.1f}% test accuracy")
print(f"ANN, shuffled cue:    {accuracy(p_no_cue,y_test):.1f}% test accuracy")
print(f"Conflicting trials: {conflict.sum()} of {n_test} held-out trials")
print(f"Linear, conflicts:  {accuracy(p_linear[conflict],y_test[conflict]):.1f}%")
print(f"ANN, conflicts:     {accuracy(p_hidden[conflict],y_test[conflict]):.1f}%")
print(f"Shuffled cue, conflicts:  {accuracy(p_no_cue[conflict],y_test[conflict]):.1f}%")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
axes[0].plot(np.arange(len(train_loss))*20, train_loss)
axes[0].set(xlabel="Training epoch", ylabel="Cross-entropy loss",
            title="Nonlinear network learns the task")
axes[0].grid(alpha=.25)
labels = ['One linear unit','Hidden-layer ANN','ANN, shuffled task cue']
all_acc = [accuracy(p_linear, y_test), accuracy(p_hidden,y_test),
           accuracy(p_no_cue,y_test)]
conf_acc = [accuracy(p_linear[conflict],y_test[conflict]),
            accuracy(p_hidden[conflict],y_test[conflict]),
            accuracy(p_no_cue[conflict],y_test[conflict])]
x = np.arange(3)
axes[1].bar(x-.18, all_acc, width=.35, label='All trials')
axes[1].bar(x+.18, conf_acc, width=.35, label='Conflicting evidence')
axes[1].axhline(50,ls=':',c='gray',lw=1)
axes[1].set(ylabel="Held-out accuracy (%)",ylim=(0,105),title="Why task context matters")
axes[1].set_xticks(x, labels, rotation=15, ha='right')
axes[1].legend()
fig.tight_layout()
plt.show()

# CLASSROOM QUESTIONS:
# 1. When motion and colour disagree, can one linear neuron follow the cue?
# 2. Is the advantage caused by stronger sensory evidence or nonlinear computation?
# 3. What changes if we shuffle the task cue across test trials?
# 4. Does this feedforward ANN reproduce the recurrent neural dynamics in Mante et al. (2013)? No.
