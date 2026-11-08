# Spike-timing-dependent plasticity (STDP) — classroom simulation

Companion to **Bi & Poo (1998)**, *Synaptic modifications in cultured hippocampal neurons: dependence on spike timing, synaptic strength, and postsynaptic cell type*, *Journal of Neuroscience* 18:10464–10472. https://doi.org/10.1523/JNEUROSCI.18-24-10464.1998

## Files
- `stdp_demo.py`: Python model (requires NumPy and Matplotlib).
- `stdp_demo.m`: same model in MATLAB.
- `stdp_model_results.png`: sample output from the Python model.

## Question
If the presynaptic and postsynaptic neurons fire the **same number of times**, can reversing their spike order change the direction of synaptic plasticity?

## Minimal rule
Let `dt = t_post - t_pre` (in milliseconds). Each paired event changes the connection by:

- `dt > 0` (pre before post): `dw = A_plus * exp(-dt/tau_plus)`.
- `dt < 0` (post before pre): `dw = -A_minus * exp(dt/tau_minus)`.
- `dt = 0`: `dw = 0` (an illustrative convention; not an empirical claim).

The code applies the increment repeatedly for `N_pairs` and bounds the weight between 0 and 1. In the example, both neurons always produce 60 paired spikes; **only their relative timing changes**.

## Manipulations
1. Compare offsets `-10`, `+10`, `-40`, `+40` ms.
2. Change `TAU_PLUS` and `TAU_MINUS` separately to test asymmetry.
3. Change `A_PLUS` and `A_MINUS` to alter potentiation/depression balance.
4. Change `N_PAIRS` and the initial weight `W0`.
5. Challenge: compare an unbounded additive rule with bounded/weight-dependent rules.

## Interpretation and limitations
This is a **phenomenological toy model**, not fitted to the observations in Bi and Poo's Figure 7. Their measurements were changes in experimentally recorded EPSC amplitudes, 20–30 min after the pairing protocol. Here weights are dimensionless model variables. The model does not simulate ion channels, membrane voltages, spontaneous spikes, stochasticity, synaptic interactions, or neuronal network dynamics. Its temporal window is illustrative, not proof of a universal STDP rule.

## Run
Python: `pip install numpy matplotlib`, then `python stdp_demo.py`.
MATLAB: Open `stdp_demo.m` and click Run.

For teaching: Show the original Figure 7 first, ask students to predict the sign of the weight change, then run the simulations and discuss which features match the experimental pattern and which are assumptions.
