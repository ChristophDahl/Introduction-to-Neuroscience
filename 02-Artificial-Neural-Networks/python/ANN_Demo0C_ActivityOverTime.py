"""Time-varying artificial-unit responses. No learning rule is implemented yet."""
import numpy as np
import matplotlib.pyplot as plt

t = np.arange(1, 121)
x = 0.75 + 0.25 * np.sin(2 * np.pi * (t - 1) / 24)
b = -1.20
w_a = np.where(t >= 61, 2.40, 0.80)
w_b = np.full_like(t, 0.80, dtype=float)
z_a, z_b = w_a * x + b, w_b * x + b
functions = [
    ("Sigmoid (0 to 1)", lambda z: 1 / (1 + np.exp(-z))),
    ("tanh (-1 to 1)", np.tanh),
    ("ReLU", lambda z: np.maximum(0, z)),
    ("Leaky ReLU", lambda z: np.maximum(0.1*z, z)),
]
fig, axs = plt.subplots(3, 1, figsize=(10, 7.2), sharex=True, constrained_layout=True)
axs[0].plot(t, x, color='black', linewidth=1.8)
axs[0].set_ylabel('Input x(t)')
axs[0].set_title('Same input, one prescribed weight change')
axs[1].plot(t, w_a, color='#2563eb', label='A: changed weight', lw=2)
axs[1].plot(t, w_b, color='#d97706', ls='--', label='B: fixed weight', lw=1.8)
axs[1].set_ylabel('Weight w(t)')
axs[1].legend(loc='upper left', fontsize=9)
axs[2].plot(t, z_a, color='#2563eb', lw=1.8)
axs[2].plot(t, z_b, color='#d97706', ls='--', lw=1.7)
axs[2].axhline(0, color='#777777', ls=':', lw=1)
axs[2].set_ylabel('Weighted sum z(t)')
axs[2].set_xlabel('Simulation step')
for ax in axs:
    ax.grid(alpha=0.2)
    ax.set_xlim(1, 120)
fig.savefig('ANN_Demo0C_InputWeights.pdf', bbox_inches='tight')

fig, axs = plt.subplots(2, 2, figsize=(11, 6.7), sharex=True, constrained_layout=True)
for ax, (name, f) in zip(axs.flat, functions):
    ax.plot(t, f(z_a), color='#2563eb', label='Changed weight', lw=2)
    ax.plot(t, f(z_b), color='#d97706', ls='--', label='Fixed weight', lw=1.8)
    ax.axvline(61, color='#666666', ls=':', lw=1)
    ax.set_title(name, fontsize=13)
    ax.set_ylabel('Output y(t)')
    ax.set_xlabel('Simulation step')
    ax.grid(alpha=0.2)
    ax.set_xlim(1, 120)
axs.flat[0].legend(loc='upper left', fontsize=9)
fig.savefig('ANN_Demo0C_ActivationOutputs.pdf', bbox_inches='tight')
print('t=60 and t=61:')
for idx in (59, 60):
    print(f't={t[idx]} x={x[idx]:.3f} wA={w_a[idx]:.1f} '
          f'zA={z_a[idx]:.3f} sigmoid={functions[0][1](z_a[idx]):.3f} '
          f'ReLU={functions[2][1](z_a[idx]):.3f}')
plt.show()
