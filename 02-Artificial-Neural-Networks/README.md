# Introduction to Artificial Neural Networks (ANNs)

**Introduction to Neuroscience — 24 November 2026 — two 50-minute sessions**

This folder contains **matched Python and MATLAB versions** of three classroom demonstrations accompanying the slide deck. Students can use either language. The models, learning rules, hyperparameters, and conceptual exercises are the same in both; the randomly generated sensory trials in Demo 1 **will not be numerically identical** between languages, even with the same random seed.

## Start here

| Activity | What you learn | MATLAB (base MATLAB) | Python (NumPy / Matplotlib) |
| --- | --- | --- | --- |
| **Demo 0: One neuron** | Weighted input, bias, sigmoid output; **no learning yet** | [`matlab/ANN_Demo0_OneNeuron.m`](matlab/ANN_Demo0_OneNeuron.m) | [`python/ANN_Demo0_OneNeuron.py`](python/ANN_Demo0_OneNeuron.py) |
| **Demo 1: Sensory decisions** | Supervised error-correcting learning, noise, held-out testing | [`matlab/ANN_Demo1_SensoryDecision.m`](matlab/ANN_Demo1_SensoryDecision.m) | [`python/ANN_Demo1_SensoryDecision.py`](python/ANN_Demo1_SensoryDecision.py) |
| **Demo 2: XOR** | Why a hidden nonlinear layer matters | [`matlab/ANN_Demo2_XOR.m`](matlab/ANN_Demo2_XOR.m) | [`python/ANN_Demo2_XOR.py`](python/ANN_Demo2_XOR.py) |

**Slides:** [`lecture/ANN_Neuroscience_2x50_2026.pdf`](lecture/ANN_Neuroscience_2x50_2026.pdf). The editable Beamer source is [`lecture/ANN_Neuroscience_2x50_2026.tex`](lecture/ANN_Neuroscience_2x50_2026.tex). The slides show MATLAB as the live-demo language; the paired Python files reproduce the same demonstrations.

**Python notebook:** [`notebook/ANN_Student_Demos.ipynb`](notebook/ANN_Student_Demos.ipynb) (standalone, suitable for Jupyter or Google Colab; contains the same Python code).

## MATLAB instructions

No additional toolboxes are needed. In MATLAB, open and run scripts in `matlab/` in order 0, 1, 2. Each `.m` file is independent.

```matlab
run('matlab/ANN_Demo0_OneNeuron.m')
run('matlab/ANN_Demo1_SensoryDecision.m')
run('matlab/ANN_Demo2_XOR.m')
```

## Python instructions

Python 3.9+ is recommended. In a terminal from this topic directory:

```bash
python -m pip install -r python/requirements.txt
python python/ANN_Demo0_OneNeuron.py
python python/ANN_Demo1_SensoryDecision.py
python python/ANN_Demo2_XOR.py
```

Or open the notebook in Colab. **No TensorFlow, PyTorch, or scikit-learn is required** for these paired scripts.

## In-class sequence (2 × 50 minutes)

**Session I:** biological inspiration → artificial neuron → weighted inputs, bias and activation → Demo 0 → gradient-based learning → Demo 1 (vary sensory noise and evaluate on held-out trials).

**Session II:** limitation of one linear neuron → XOR → nonlinear hidden units → idea of backpropagation → Demo 2 → biological comparison and limits of ANNs.

## Exercise suggestions

- **Demo 0:** Swap the left and right inputs. What happens to the decision? Now change the bias.
- **Demo 1:** Try noise = 0.2, 0.6, and 1.2. What happens to test accuracy? What do the signs of the weights mean? Why test on *unseen* trials?
- **Demo 2:** Why can a single straight decision boundary not classify XOR? Why does the trained nonlinear network succeed? Does success on four *training* cases prove generalization?

## Scientific caveats

1. The sensory trials are **synthetic Gaussian inputs**, not recordings from MT or LIP. The connection to Law and Gold (2008) is conceptual (sensory encoding versus decision readout).
2. Supervised gradient descent and backpropagation are **not** equivalent to biological Hebbian LTP/LTD.
3. The four XOR examples are both the training and evaluation data; this demonstration establishes representational capacity, **not** generalization performance.
4. ANN hidden units and layers should not be interpreted as anatomical layers of cortex.
5. Numerical results in Demo 1 differ between MATLAB and NumPy because their random-number generators are different; comparison should focus on the learning pattern, not identical random trials.

## Optional visuals

- [3Blue1Brown: But what is a neural network?](https://www.3blue1brown.com/lessons/neural-networks/)
- [TensorFlow Playground](https://playground.tensorflow.org/)
