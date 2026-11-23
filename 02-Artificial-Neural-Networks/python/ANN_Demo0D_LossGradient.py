"""One-dimensional gradient descent using the same numerical example as slide 13."""
import numpy as np
import matplotlib.pyplot as plt

target, x, eta, n_steps = 2.0, 1.0, 0.5, 7
weights = [0.0]
for k in range(n_steps):
    y = weights[-1] * x
    grad = (y - target) * x
    weights.append(weights[-1] - eta * grad)
losses = [0.5*(target - w*x)**2 for w in weights]
for i,(w,loss) in enumerate(zip(weights,losses)):
    print(f"Step {i}: w={w:.4f}, y={w*x:.4f}, loss={loss:.4f}")
ww = np.linspace(-0.5,3.5,401)
plt.figure(figsize=(8,5))
plt.plot(ww,0.5*(target-ww*x)**2,label='Loss curve',linewidth=2)
plt.plot(weights,losses,'o-',label='Successive updates',linewidth=1.8)
plt.xlabel('Weight w');plt.ylabel('Loss L');plt.title('Moving towards lower prediction error')
plt.grid(alpha=.2);plt.legend();plt.tight_layout();plt.show()
