import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import cooltools
import cooler
import cooltools.lib.plotting
from matplotlib.ticker import EngFormatter

from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import csv
import pysam 
import torch
from torch.utils.data import DataLoader,Dataset
from torch import nn
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.autograd import Variable

train,test=torch.load("./data/train.pth"),torch.load("./data/test.pth")
train_loader,test_loader= DataLoader(train, batch_size=5,shuffle=True),DataLoader(test, batch_size=5,shuffle=True)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def plot_hic(x,name="hello"):
    mat=torch.zeros(448,448)
    idx=torch.triu_indices(448, 448,2)
    mat[idx[0],idx[1]]=x.float()
    im=plt.imshow(mat+mat.T,vmax=1)
    plt.colorbar(im ,fraction=0.046, pad=0.04, label='balanced')
    plt.savefig(f"{name}.png")
    plt.show()

from typing import Sequence

class Conv1D_block(nn.Module):
  def __init__(self,in_channels,out_channels,kernel_size=11,dilation=10,m_kernel=2):
    super().__init__()
    padding=dilation*(kernel_size-1)//2
    self.conv1d=nn.Conv1d(in_channels, out_channels,kernel_size,padding=padding,dilation=dilation)
    self.relu1=nn.ReLU()
    self.bn=nn.BatchNorm1d(out_channels)
    self.pool=nn.MaxPool1d(m_kernel,stride=2)
    self.Seq=nn.Sequential(self.relu1,self.conv1d,self.bn,self.pool)
  def __call__(self,x):
    return self.Seq(x)

class ResNet1d(nn.Module):
  def __init__(self,in_channels,out_channels,kernel_size=5,dilation=10):
    super().__init__()
    padding=dilation*(kernel_size-1)//2
    self.conv1d1,self.conv1d2=nn.Conv1d(in_channels, out_channels//2,kernel_size,padding=padding,dilation=dilation),nn.Conv1d(out_channels//2, in_channels,kernel_size,padding=padding,dilation=dilation)
    self.relu1,self.relu2=nn.ReLU(),nn.ReLU()
    self.Dropout=nn.Dropout()
    self.bn1,self.bn2=nn.BatchNorm1d(out_channels//2),nn.BatchNorm1d(in_channels)
    self.Seq=nn.Sequential(self.relu1,self.conv1d1,self.bn1,self.relu2,self.conv1d2,self.bn2,self.Dropout)
  def __call__(self,x):
    y=x
    return self.Seq(y)+x
class one_two(nn.Module):
  def __init__(self,in_channels, out_channels,mid_channel=64,kernel_size=5,dilation=10,dim=512,device="cpu"):
    super().__init__()
    padding=dilation*(kernel_size-1)//2
    self.conv1d=nn.Conv1d(in_channels, mid_channel ,kernel_size,padding=padding,dilation=dilation)
    self.relu1,self.relu2,self.relu3=nn.ReLU(),nn.ReLU(),nn.ReLU()
    self.bn,self.bn1=nn.BatchNorm1d(mid_channel),nn.BatchNorm2d(out_channels)
    self.Seq=nn.Sequential(self.relu1,self.conv1d,self.bn,self.relu2)
    self.conv2d,self.conv2d1=nn.Conv2d(in_channels, mid_channel+1,1),nn.Conv2d(mid_channel+1,out_channels ,kernel_size=5,padding=2)
    self.sem=symmetrize2d()
    self.Seq2=nn.Sequential(self.relu3,self.conv2d1,self.bn1,self.sem)
    l=dim
    a=torch.tensor([list(range(l))])
    b=a.mT
    self.dist=torch.abs(b-a).unsqueeze(0).unsqueeze(0)
    self.dist=self.dist.to(device)
  def __call__(self, x):
    x=self.Seq(x)
    y=torch.cat([torch.add(x.unsqueeze(-1),torch.transpose(x.unsqueeze(-1), -2, -1))/2,self.dist.repeat(x.shape[0],1,1,1)],1)
    return self.Seq2(y)

class conv2d_block(nn.Module):
  def __init__(self,in_channels,out_channels,kernel_size,dilation=5):
    super().__init__()
    padding=dilation*(kernel_size-1)//2
    self.conv2d=nn.Conv1d(in_channels, out_channels,kernel_size,padding=padding,dilation=dilation)
    self.relu=nn.ReLU()
    self.bn1=nn.BatchNorm2d(out_channels)
    self.Seq=nn.Sequential(self.relu1,self.conv2d,self.bn1)

  def __call__(self,x):
    x=self.Seq(x)
    return x
class symmetrize2d(nn.Module):
  def __init__(self):
    super().__init__()
  def __call__(self,x):
    return (x+x.transpose(-1,-2))/2

class ResNet2d(nn.Module):
  def __init__(self,in_channels,out_channels,kernel_size=3,dilation=5):
    super().__init__()
    padding=dilation*(kernel_size-1)//2
    self.conv2d1,self.conv2d2=nn.Conv2d(in_channels, out_channels//2,kernel_size,padding=padding,dilation=dilation),nn.Conv2d(out_channels//2, in_channels,kernel_size,padding=padding,dilation=dilation)
    self.relu1,self.relu2=nn.ReLU(),nn.ReLU()
    self.Dropout=nn.Dropout()
    self.bn1,self.bn2=nn.BatchNorm2d(out_channels//2),nn.BatchNorm2d(in_channels)
    self.Seq=nn.Sequential(self.relu1,self.conv2d1,self.bn1,self.relu2,self.conv2d2,self.bn2,self.Dropout)
  def __call__(self,x):
    y=x
    return self.Seq(y)+x
class Crop2D(nn.Module):
  def __init__(self,length):
    super().__init__()
    self.length=length
  def __call__(self,x):
    return x[:,:,self.length:-self.length,self.length:-self.length]
class Upper_triangle(nn.Module):
  def __init__(self):
    super().__init__()
  def __call__(self,x):
    return x[:,:,torch.triu(torch.ones(x.shape[-1], x.shape[-1]),2) == 1]
class FC(nn.Module):
  def __init__(self,in_f,out_f=5):
    super().__init__()
    self.fc=nn.Linear(in_f,out_f)
  def __call__(self,x):
    return self.fc(x.transpose(-2,-1))

class Model(nn.Module):
  def __init__(self,in_channels,conv1D_block=11,resNet1d=5,resNet2d=5,out=1,device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    super().__init__()
    layers=[Conv1D_block(in_channels,96)]
    for i in range(conv1D_block-1):
      layers.append(Conv1D_block(96,96))
    for i in range(resNet1d):
      layers.append(ResNet1d(96,96))
    layers.append(one_two(96,48,device=device))
    for i in range(resNet2d):
      layers.append(ResNet2d(48,48))
      layers.append(symmetrize2d())
    layers.append(Crop2D(32))
    layers.append(Upper_triangle())
    layers.append(FC(48,out))
    self.layers=layers
    self.Seq=nn.Sequential(*self.layers)
  def __call__(self,x):
    # for i in self.layers:
    #   print(x.shape)
    #   x=i(x)
    return self.Seq(x)


def train(net, optimizer, criterion,train_loader,epochs,root="./",model_name="Akita", plot=False,device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    model = net.to(device)
    
    total_step = len(train_loader)
    overall_step = 0
    train_loss_values = []
    for epoch in range(epochs):
        total = 0
        running_loss = 0.0
        for i, (X, y) in enumerate(train_loader):
            # Move tensors to configured device
            X = X.to(device).float()
            y = y.to(device).float()
            #Forward Pass
            outputs = model(X)
            loss = criterion(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            running_loss += loss.item()
            total += y.size(0)
            optimizer.step()
            if (i+1) % 4 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, i+1, total_step, loss.item()))
            if plot:
                info = { ('loss_' + model_name): loss.item() }
        train_loss_values.append(running_loss/total)
        if running_loss/total<1e-3 : break
    return train_loss_values

def test(net,criterion,test_loader,device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    model = net.to(device)
    model.eval()   
    with torch.no_grad():
        total = 0
        running_loss=0.0
        for i, (X, y) in enumerate(test_loader):
            X = X.to(device).float()
            y = y.to(device).float()
            outputs = model(X)
            loss = criterion(outputs, y)
            running_loss += loss.item()
            total += y.size(0)
            
        print('Accuracy of the network on the test: {}'.format(running_loss/total))

net=Model(5)
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001)
epochs=1
#net, optimizer, criterion,filename,epochs
train_loss_values=train(net, optimizer, criterion,train_loader, epochs,model_name="Akita")
torch.save(net.state_dict(), "./model.pth")

plt.plot(list(range(len(train_loss_values))),train_loss_values,label="training error",color="red")
plt.legend()
plt.savefig("trainingcurve.png")
plt.show()
print(train_loss_values[-1])

test(net,criterion,test_loader)

