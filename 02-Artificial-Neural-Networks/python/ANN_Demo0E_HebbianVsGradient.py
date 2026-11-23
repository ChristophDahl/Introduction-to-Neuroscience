"""Demo 0E: Hebbian versus gradient updates on identical inputs.

NumPy and Matplotlib. Hebbian rule is intentionally unregulated and does
not use the teaching target. The gradient rule minimises squared error.
"""
import numpy as np
import matplotlib.pyplot as plt


def run(n_steps=140, eta_h=0.040, eta_g=0.140):
    k = np.arange(n_steps)
    x = 0.65 + 0.30 * np.sin(k / 11) + 0.10 * np.cos(k / 4)
    target_gain = 1.40
    target = target_gain * x
    w_h = np.zeros(n_steps + 1)
    w_g = np.zeros(n_steps + 1)
    w_h[0] = w_g[0] = 0.40
    y_h, y_g = np.zeros(n_steps), np.zeros(n_steps)
    loss_h, loss_g = np.zeros(n_steps), np.zeros(n_steps)
    for i in range(n_steps):
        y_h[i], y_g[i] = w_h[i] * x[i], w_g[i] * x[i]
        loss_h[i] = 0.5 * (target[i] - y_h[i]) ** 2
        loss_g[i] = 0.5 * (target[i] - y_g[i]) ** 2
        w_h[i+1] = w_h[i] + eta_h * x[i] * y_h[i]
        grad = (y_g[i] - target[i]) * x[i]
        w_g[i+1] = w_g[i] - eta_g * grad
    fig, axs = plt.subplots(2, 2, figsize=(11, 6.5))
    axs[0,0].plot(k+1, x, 'k', label='Input x')
    axs[0,0].plot(k+1, target, label='Target (gradient only)')
    axs[0,0].set_title('Same input stream'); axs[0,0].legend()
    axs[0,1].plot(np.arange(n_steps+1), w_h, label='Hebbian')
    axs[0,1].plot(np.arange(n_steps+1), w_g, label='Gradient')
    axs[0,1].axhline(target_gain, ls=':', color='k', label='Target gain')
    axs[0,1].set_title('Weight trajectories'); axs[0,1].legend()
    axs[1,0].plot(k+1, target, 'k:', label='Target')
    axs[1,0].plot(k+1, y_h, label='Hebbian output')
    axs[1,0].plot(k+1, y_g, label='Gradient output')
    axs[1,0].set_title('Outputs'); axs[1,0].legend()
    axs[1,1].plot(k+1, loss_h, label='Hebbian evaluated against target')
    axs[1,1].plot(k+1, loss_g, label='Gradient loss')
    axs[1,1].set_title('Squared error'); axs[1,1].legend()
    for ax in axs.flat:
        ax.grid(alpha=.2); ax.set_xlabel('Training step')
    fig.tight_layout()
    print(f'Final weights: Hebbian={w_h[-1]:.3f}, gradient={w_g[-1]:.3f}')
    return fig, {'w_h':w_h,'w_g':w_g,'y_h':y_h,'y_g':y_g,'loss_h':loss_h,'loss_g':loss_g}


if __name__ == '__main__':
    run()
    plt.show()
