"""ANN Demo 0B: change weights and bias while keeping sensory trials fixed.

Same CSV sample bank as the MATLAB version and the three panels on slide 10.
Requires NumPy and Matplotlib; uses no machine-learning framework.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

n_samples = 160          # try 30, 160, 500
sensory_noise = 0.18     # try .08, .18, .35

bank = np.loadtxt(Path(__file__).resolve().parent.parent / 'data' /
                  'ANN_demo0B_trials.csv', delimiter=',', skiprows=1)
assert 2 <= n_samples <= len(bank)
labels = bank[:n_samples, 0].astype(int)
X = np.column_stack((.72*(1-labels)+.24*labels+sensory_noise*bank[:n_samples,1],
                     .24*(1-labels)+.72*labels+sensory_noise*bank[:n_samples,2]))
X = np.clip(X, 0, 1)

cases = [
    ('Baseline', (-2,2), 0),
    ('Relative weights', (-1,3), 0),
    ('Positive bias', (-2,2), .8),
    ('Large weights', (-6,6), 0),
    ('Small weights', (-.5,.5), 0),
    ('Reverse signs', (2,-2), 0),
]


def sigmoid(z):
    return 1/(1+np.exp(-np.clip(z,-30,30)))


def plot_case(ax,name,weights,bias):
    w=np.asarray(weights)
    p=sigmoid(X@w+bias)
    pred=(p>=.5).astype(int)
    accuracy=100*np.mean(pred==labels)
    ax.scatter(*X[labels==0].T,s=15,c='#10959E',label='Feature 1')
    ax.scatter(*X[labels==1].T,s=15,c='#E8733B',label='Feature 2')
    wrong=pred!=labels
    ax.scatter(*X[wrong].T,s=40,c='black',marker='x',linewidth=.8,label='Error')
    xx=np.linspace(0,1,200)
    if abs(w[1]) > 1e-9:
        yy=-(w[0]*xx+bias)/w[1]
        ax.plot(xx,yy,'k--',lw=1.8)
    else:
        ax.axvline(-bias/w[0],ls='--',c='black')
    ax.set(xlim=(0,1),ylim=(0,1),xlabel='Feature 1 input',ylabel='Feature 2 input',
           title=f'{name}: {accuracy:.1f}%')
    ax.set_aspect('equal')
    ax.grid(alpha=.2)
    print(f'{name:18s} weights={tuple(w)!s:14s} bias={bias:4.1f} accuracy={accuracy:5.1f}%')


fig,axs=plt.subplots(1,3,figsize=(11.5,3.8))
for ax,(name,weights,bias) in zip(axs,cases[:3]): plot_case(ax,name,weights,bias)
fig.suptitle('Same 160 sensory trials; different weights/bias')
fig.tight_layout()

fig,axs=plt.subplots(1,3,figsize=(11.5,3.8))
for ax,(name,weights,bias) in zip(axs,cases[3:]): plot_case(ax,name,weights,bias)
fig.suptitle('Scaling versus reversing weights')
fig.tight_layout()

fig,ax=plt.subplots(figsize=(6,3.9))
d=np.linspace(-1,1,401)
for scale in [.5,2,6]:
    ax.plot(d,sigmoid(scale*d),label=f'Weight magnitude {scale}')
ax.axhline(.5,color='gray',ls=':')
ax.set(xlabel='Evidence difference: feature 2 − feature 1',ylabel='Sigmoid output',
       title='Same boundary, different sigmoid steepness')
ax.legend()
ax.grid(alpha=.2)
fig.tight_layout()
plt.show()
