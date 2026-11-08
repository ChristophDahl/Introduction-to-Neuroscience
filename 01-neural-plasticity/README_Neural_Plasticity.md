# Neural Plasticity — MATLAB demonstrations

This folder accompanies the **Neural Plasticity** lectures in *Introduction to Neuroscience*.

## 1. Hebbian learning: paired vs unpaired activity

**File:** [`hebbian_learning_demo.m`](hebbian_learning_demo.m)

A simple simulation of the Hebbian update rule:

\[
w(t+1)=w(t)+\eta\,x(t)y(t)
\]

Here, `x` and `y` represent binary pre- and postsynaptic activity, `eta` is the learning rate, and `w` is synaptic weight.

The code compares two conditions with the same individual activity frequencies:

- **Paired:** both neurons are active on the same trials.
- **Unpaired:** the neurons are active on alternating trials.

**Try this:** Before running the program, predict the change in weight in each condition. Then change `eta`, `N`, or `w0` and rerun.

## 2. Systematic Hebbian parameter sweep

**File:** [`hebbian_parameter_sweep.m`](hebbian_parameter_sweep.m)

This script evaluates **90 parameter combinations** using a factorial design:

| Parameter | Meaning | Values |
| --- | --- | --- |
| `eta` | Learning rate | 0.005, 0.020, 0.050 |
| `p` | Activity fraction of each neuron | 0.20, 0.40 |
| `c` | Co-activation: P(y=1 given x=1) | 0, 0.25, 0.50, 0.75, 1 |
| `N` | Number of trials | 40, 80, 160 |
| `w0` | Initial synaptic weight | 0.20, 0.50 |

Crucially, **both neurons have the same overall activity rate** within each condition. This helps isolate co-activation from firing rate.

For the controlled binary activity patterns used here, the final weight is:

\[
w_{\mathrm{final}}=w_0+\eta Npc
\]

**Try this:** Predict which parameters change the *amount of strengthening* and which merely shift the final weight. Use the output graphs to check your predictions.

## Running the MATLAB scripts

1. Download the `.m` files, or download the repository using **Code → Download ZIP**.
2. In MATLAB, navigate to the folder containing the scripts.
3. Open a script and click **Run**, or type its name (without `.m`) in the Command Window.
4. Inspect the figures and, for the parameter sweep, the generated `hebbian_outputs` folder. The sweep saves two figures and a `hebbian_factorial_results.csv` file.

The parameter-sweep script is intended for **MATLAB R2016b or later**. Neither script requires additional toolboxes.

## Questions for discussion

1. Why does paired activity strengthen the synapse but completely non-overlapping activity does not?
2. Is co-activation the same as statistical correlation or a causal relationship?
3. What happens if the two neurons fire independently but sometimes coincide?
4. Why do weights increase without bound in the simple model?
5. What additional processes would be needed to model LTD, spike timing, or homeostatic regulation?

## Important limitations

These are intentionally minimal **teaching models**, not measured hippocampal data. The update is additive and uses nonnegative binary activity. It does **not** include synaptic weakening (LTD), weight saturation, precise spike timing (STDP), or homeostatic regulation. For this rule, independent activity with positive firing rates can still cause strengthening; the `c = 0` condition means *no coincidences*, not statistical independence.
