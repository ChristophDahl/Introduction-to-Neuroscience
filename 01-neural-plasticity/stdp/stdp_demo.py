"""STDP classroom demo following the discussion of Bi & Poo (1998).

A deliberately simple, phenomenological learning rule, not a fit to the
experimental measurements. Requires numpy and matplotlib.
"""
import numpy as np
import matplotlib.pyplot as plt

# ---------- Parameters to experiment with ----------
N_PAIRS = 60           # number of repeated pre/post spike pairings
W0 = 0.40              # initial weight, scaled between 0 and 1
W_MIN, W_MAX = 0.0, 1.0
A_PLUS = 0.005         # potentiation per near-coincident pairing
A_MINUS = 0.003        # depression per near-coincident pairing
TAU_PLUS = 15.0        # potentiation time window (ms)
TAU_MINUS = 15.0       # depression time window (ms)
# ---------- End user parameters ----------


def update_per_pair(dt_ms, a_plus=A_PLUS, a_minus=A_MINUS,
                    tau_plus=TAU_PLUS, tau_minus=TAU_MINUS):
    """Weight increment per pairing, with dt = post_time - pre_time (ms)."""
    d = np.asarray(dt_ms, dtype=float)
    return np.where(d > 0, a_plus * np.exp(-np.maximum(d, 0) / tau_plus),
                    np.where(d < 0, -a_minus * np.exp(np.minimum(d, 0) / tau_minus), 0.0))


def simulate(dt_ms, n_pairs=N_PAIRS, w0=W0):
    """Repeated spike pairs at a fixed temporal offset, with bounded weights."""
    weights = np.zeros(n_pairs + 1)
    weights[0] = w0
    change = float(update_per_pair(dt_ms))
    for trial in range(n_pairs):
        weights[trial + 1] = np.clip(weights[trial] + change, W_MIN, W_MAX)
    return weights


def main():
    dt = np.linspace(-60, 60, 481)
    final_weights = np.array([simulate(d)[-1] for d in dt])
    deltas = 100 * (final_weights - W0) / W0
    trials = np.arange(N_PAIRS + 1)

    fig, ax = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    # A: STDP timing window after 60 pairings.
    ax[0, 0].plot(dt, deltas, linewidth=2)
    ax[0, 0].axvline(0, color='0.5', linewidth=1)
    ax[0, 0].axhline(0, color='0.5', linestyle='--', linewidth=1)
    ax[0, 0].set(xlabel='Spike timing: t_post - t_pre (ms)',
                 ylabel='Change in weight (%)',
                 title='A. Timing determines potentiation or depression')

    # B: all conditions have the same number of spikes.
    for delay in (10, -10, 40, -40):
        ax[0, 1].plot(trials, simulate(delay), label=f'dt = {delay:+d} ms', linewidth=2)
    ax[0, 1].set(xlabel='Pairing number', ylabel='Synaptic weight',
                 title='B. Same number of spikes, different order')
    ax[0, 1].legend(frameon=False)

    # C: Width of STDP timing window.
    for tau in (5, 15, 30):
        per_pair = update_per_pair(dt, tau_plus=tau, tau_minus=tau)
        # No saturation for the plotted parameters; clipping retains physical bounds.
        final = np.clip(W0 + N_PAIRS * per_pair, W_MIN, W_MAX)
        ax[1, 0].plot(dt, 100 * (final - W0) / W0, label=f'tau = {tau} ms')
    ax[1, 0].axhline(0, color='0.5', linestyle='--', linewidth=1)
    ax[1, 0].set(xlabel='Spike timing (ms)', ylabel='Change in weight (%)',
                 title='C. Time-window width changes sensitivity')
    ax[1, 0].legend(frameon=False)

    # D: Number of repeated pairings.
    for n in (10, 30, 60, 100):
        ax[1, 1].plot(dt,
                      100 * (np.clip(W0 + n * update_per_pair(dt), W_MIN, W_MAX) - W0) / W0,
                      label=f'N = {n}')
    ax[1, 1].axhline(0, color='0.5', linestyle='--', linewidth=1)
    ax[1, 1].set(xlabel='Spike timing (ms)', ylabel='Change in weight (%)',
                 title='D. Repetition changes the magnitude')
    ax[1, 1].legend(frameon=False)

    fig.suptitle('STDP — a minimal model, not fitted to Bi & Poo (1998)', fontsize=15)
    fig.savefig('stdp_model_results.png', dpi=180)
    plt.show()
    print('After 60 pairings (initial weight 0.40):')
    for d in (-40, -10, 10, 40):
        print(f'  dt = {d:+3d} ms: final weight = {simulate(d)[-1]:.3f}')


if __name__ == '__main__':
    main()
