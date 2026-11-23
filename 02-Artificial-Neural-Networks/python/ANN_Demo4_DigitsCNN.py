"""Demo 4: same bundled 8x8 handwritten digit database used by MATLAB.

Requires: numpy, scipy, matplotlib, torch, scikit-learn.
Runs offline. Produces 20 predictions and a confusion matrix.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset

np.random.seed(27);torch.manual_seed(27)
torch.set_num_threads(2)
path=Path(__file__).resolve().parent.parent/'data'/'ANN_digits8x8.mat'
a=loadmat(path)
images=a['images'].squeeze(axis=2).transpose(2,0,1).astype('float32')/16
labels=a['labels'].reshape(-1).astype('int64')
ixTrain,ixTest=train_test_split(np.arange(len(labels)),test_size=.20,
                                   random_state=27,stratify=labels)
Xtrain=torch.tensor(images[ixTrain,None,:,:]);Ytrain=torch.tensor(labels[ixTrain])
Xtest=torch.tensor(images[ixTest,None,:,:]);Ytest=torch.tensor(labels[ixTest])
trainloader=DataLoader(TensorDataset(Xtrain,Ytrain),batch_size=64,shuffle=True)

net=nn.Sequential(nn.Conv2d(1,8,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
                  nn.Conv2d(8,16,3,padding=1),nn.ReLU(),nn.Flatten(),
                  nn.Linear(16*4*4,10))
opt=torch.optim.Adam(net.parameters(),lr=.003);criterion=nn.CrossEntropyLoss()
for epoch in range(12):
    net.train();summ=0
    for xb,yb in trainloader:
        opt.zero_grad();loss=criterion(net(xb),yb);loss.backward();opt.step()
        summ+=loss.item()*len(yb)
    print(f'Epoch {epoch+1:2d}: training loss={summ/len(Ytrain):.4f}')
net.eval()
with torch.no_grad():pred=net(Xtest).argmax(1).numpy()
accuracy=(pred==labels[ixTest]).mean()
print(f'Held-out digit accuracy: {100*accuracy:.1f}% ({len(ixTest)} images)')
fig,axs=plt.subplots(2,10,figsize=(12,3.5))
for k,ax in enumerate(axs.flat):
    ax.imshow(images[ixTest[k]],cmap='gray_r',interpolation='nearest',vmin=0,vmax=1)
    ax.set_title(f'{pred[k]} / {labels[ixTest[k]]}',
                 color='forestgreen' if pred[k]==labels[ixTest[k]] else 'firebrick')
    ax.axis('off')
fig.suptitle(f'Predicted / actual digit | held-out accuracy {accuracy:.1%}')
fig.tight_layout();fig.savefig(Path(__file__).resolve().parent.parent/'lecture'/'figures'/'ANN_Digits_CNN_Predictions_Python.png',dpi=165);plt.show()
fig,ax=plt.subplots(figsize=(5,4.5))
cm=confusion_matrix(labels[ixTest],pred,labels=np.arange(10))
im=ax.imshow(cm,cmap='Blues');fig.colorbar(im,ax=ax)
ax.set(xlabel='Predicted digit',ylabel='Actual digit',title='Held-out confusion matrix',
       xticks=np.arange(10),yticks=np.arange(10))
fig.tight_layout();fig.savefig(Path(__file__).resolve().parent.parent/'lecture'/'figures'/'ANN_Digits_CNN_Confusion_Python.png',dpi=165);plt.show()
