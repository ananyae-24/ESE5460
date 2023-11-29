import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import cooltools
import cooler
import cooltools.lib.plotting
from matplotlib.ticker import EngFormatter
import torchvision as thv

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

ctrl = cooler.Cooler('../data/coolers/H1hESC_hg38_4DNFI1O6IL1Q.mapq_30.2048.cool')
ref = pysam.FastaFile('../data/hg38.ml.fa')

from scipy import interpolate
from astropy.convolution import convolve

class inter():
    def __init__(self):
        x = np.arange(0, 448)
        y = np.arange(0, 448)
        self.xx, self.yy = np.meshgrid(x, y)
        self.kernel=self.gaussian_kernel(19,20)
    def gaussian_kernel(self,kernel_size, sigma):
        kernel = np.fromfunction(lambda x, y: (1 / (2 * np.pi * sigma**2)) * np.exp(-((x - kernel_size//2)**2 + (y - kernel_size//2)**2) / (2 * sigma**2)), (kernel_size, kernel_size))
        normal = kernel / np.sum(kernel)
        return normal
    def __call__(self,arr):
        d=arr.diagonal().reshape((arr.shape[0],1))
        mask=(d==0)@((d==0).T)
        arr=np.where(mask,np.nan,arr)
        newarr=convolve(arr,self.kernel)
        arr[mask==True]=newarr[mask==True]
        return np.nan_to_num(arr, nan=0.0)
        
class RandomComplement():
    def __init__(self, dna_pair_columns, random_seed, transform_prob):
        """e.g. pairs = ((0, 1), (2, 3))"""
        self.transform_prob = transform_prob
        self.dna_pair_columns = dna_pair_columns
        self.random_generator = np.random.default_rng(random_seed)

    def __call__(self, x:torch.Tensor):
        if self.random_generator.random() > self.transform_prob:
            return x

        complement = torch.zeros_like(x)
        for pair in self.dna_pair_columns:
            complement[:, :, pair[0]] = x[:, :, pair[1]]
            complement[:, :, pair[1]] = x[:, :, pair[0]]
        
        complement[:, :, 4] = x[:, :, 4]

        return complement

class WeightedMSE():
    def __init__(self):
        dim = 448
        x = torch.abs(torch.arange(dim).unsqueeze(0)-torch.arange(dim).unsqueeze(1))
        square_weights = self.diagonal_fun(x)
        self.weigths = square_weights[*torch.triu_indices(*square_weights.shape, 2)].unsqueeze(0).unsqueeze(-1)

        print(self.weigths.shape)

    def diagonal_fun(self, x, max_weight=36):
        return 1 + max_weight * torch.sin(x/500*torch.pi)
    
    def __call__(self, y_pred:torch.Tensor, y_true:torch.Tensor):
        """
        Compute weighted mean square error for prediction.
        Args:
            y_pred: Predicted labels
            y_true: True labels
        """
        return ((y_true - y_pred)**2 * self.weigths).sum()/y_true.shape[0]

RANDOM_SEED = 42 
transform = thv.transforms.Compose((
   RandomComplement(((0, 1), (2, 3)), RANDOM_SEED, 0.5)
))

class TorchDataset(Dataset):
    def __init__(self, file,fasta,cool,root_dir="./", transform=None):
        """
        Arguments:
            file (string): Path to the file with annotations.
            root_dir (string): Directory with all the tfr files.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # self.landmarks_frame = pd.read_csv(csv_file)
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        # to store dictionary for genomic data
        self.df_frame=pd.read_csv(file)
        self.transform=transform
        self.fasta=fasta
        self.cooler=cool
        self.i=inter()
    def one_hot_enc(self,seq):
        ans=[]
        for i in seq:
            if i =="A":
                ans.append([1.0,0,0,0,0])
            elif i =="T":
                ans.append([0,1.0,0,0,0])
            elif i=="G":
                ans.append([0,0,1.0,0,0])
            elif i=="C":
                ans.append([0,0,0,1.0,0])
            else:
                ans.append([0,0,0,0,1.0])

        return np.array(ans).T
    def flatten(self,mat):
        return torch.tensor(mat[np.triu_indices(mat.shape[0],2)]).unsqueeze(-1)
    def K(self,p,q,sigma=1):
        return np.exp(-((p[0]-q[0])**2+(p[1]-q[1])**2)**.5/(2*sigma**2))
    def cal_value_l(self,p,b,mat,sigma=10):
        val,weight=0,0
        for i in range(max(0,p[0]-b),min(mat.shape[0],p[0])):
            for j in range(max(0,p[1]-b),min(mat.shape[1],p[1])):
                w=self.K(p,[i,j],sigma)
                val+=mat[i,j]*w
                weight+=w
        if weight==0:
            return mat[p[0]-1][p[1]-1]
        return val/weight
    def cal_value_r(self,p,b,mat,sigma=10):
        val,weight=0,0
        for i in range(max(0,p[0]),min(mat.shape[0],p[0]+b)):
            for j in range(max(0,p[1]),min(mat.shape[1],p[1]+b)):
                w=self.K(p,[i,j],sigma)
                val+=mat[i,j]*w
                weight+=w
        if weight==0:
            return mat[p[0]+1][p[1]+1]
        return val/weight
    def fill_mat(self,mat,b=5,sigma=30):
        # mat=np.array(m, copy=True)
        d=np.diag(mat)
        for i in range(mat.shape[0]):
            if d[i]==0:
                for j in range(i,mat.shape[0]):
                    mat[i,j]=self.cal_value_l([i,j],b,mat,sigma)
                for j in range(i,-1,-1):
                    mat[i,j]=self.cal_value_l([i,j],b,mat,sigma)
                for j in range(i,mat.shape[1]):
                    mat[j,i]=self.cal_value_l([j,i],b,mat,sigma)
                for j in range(i,-1,-1):
                    mat[j,i]=self.cal_value_l([j,i],b,mat,sigma)
        for i in range(mat.shape[0]-1,-1,-1):
            if d[i]==0:
                for j in range(i,mat.shape[0]):
                    mat[i,j]=max(mat[i,j],self.cal_value_r([i,j],b,mat,sigma))
                for j in range(i,-1,-1):
                    mat[i,j]=max(mat[i,j],self.cal_value_r([i,j],b,mat,sigma))
                for j in range(i,mat.shape[1]):
                    mat[j,i]=max(mat[j,i],self.cal_value_r([j,i],b,mat,sigma))
                for j in range(i,-1,-1):
                    mat[j,i]=max(mat[j,i],self.cal_value_r([j,i],b,mat,sigma))
        return mat
    def __len__(self):
        return len(self.df_frame)
    def __getitem__(self, idx):
        curr=self.df_frame.iloc[idx,:]
        X=self.fasta.fetch(curr["name"],curr["start"],curr["end"])
        y=self.flatten(self.i(self.cooler.matrix(balance=False).fetch((curr["name"],curr["start"],curr["end"])))[32:-32,32:-32])
        word_to_ix = {"A":0,"T":1,"G":2,"C":3,"N":4 }
        X=[word_to_ix[w] for w in X]
        X = np.eye(5)[X, :]
        X=X.T
        if self.transform:
            X=self.transform(X)
        sample=(X,y)
        return sample
    
train,val,test=TorchDataset("../data/cvsDatasets/train.csv",ref,ctrl,transform),TorchDataset("../data/cvsDatasets/val.csv",ref,ctrl),TorchDataset("../data/cvsDatasets/test.csv",ref,ctrl)
train_loader,val_loader,test_loader=DataLoader(train, batch_size=2,shuffle=True),DataLoader(val, batch_size=2,shuffle=True),DataLoader(test, batch_size=2,shuffle=True)

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
    self.relu=nn.ReLU()
  def __call__(self,x):
    return self.relu(self.fc(x.transpose(-2,-1)))

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

import math
def train(net, optimizer, criterion,train_loader,val_loader,epochs,root="./",model_name="Akita", plot=False,device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    model = net.to(device)
    total_step = len(train_loader)
    overall_step = 0
    train_loss_values = []
    validation_values=[]
    T=len(train_loader)*epochs
    T0=2*T//5
    t=0
    
    for epoch in range(epochs):
        total = 0
        running_loss = 0.0
        model.train(True)
        for i, (X, y) in enumerate(train_loader):
            for op_params in optimizer.param_groups:
              if t<=T0:
                op_params['lr'] =1e-4+.001
              else:
                f=(t-T0)/(T-T0)
                op_params['lr'] =1e-6+(.001*math.cos(f*math.pi/2))
            # Move tensors to configured device
            X = X.to(device).float()
            y = y.to(device).float()
            #Forward Pass
            outputs = model(X)
            optimizer.zero_grad()
            loss = criterion(outputs, y)
            loss.backward()
            running_loss += loss.item()
            total += y.size(0)
            optimizer.step()
            t+=1
            if (i+1) % 10 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, i+1, total_step, loss.item()))
            if plot:
                info = { ('loss_' + model_name): loss.item() }
        train_loss_values.append(running_loss/total)
        validation_values.append(test(model,criterion,val_loader))
        if epoch+1==10:
           torch.save(model.state_dict(), "./model_10.pth")
        if epoch+1==15:
           torch.save(model.state_dict(), "./model_15.pth")
        if epoch+1==20:
           torch.save(model.state_dict(), "./model_20.pth")
        if running_loss/total<=1e-3 : break
    return (train_loss_values,validation_values)

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
        return running_loss/total

net=Model(5)
criterion = WeightedMSE()
optimizer = optim.SGD(net.parameters(), lr=0.001)
epochs=30
#net, optimizer, criterion,filename,epochs
train_loss_values,validatio_loss=train(net, optimizer, criterion,train_loader,val_loader, epochs,model_name="Akita")
torch.save(net.state_dict(), "./model.pth")

plt.plot(list(range(len(train_loss_values))),train_loss_values,label="training error",color="red")
plt.legend()
plt.savefig("trainingcurve.png")
plt.show()

plt.plot(list(range(len(validatio_loss))),validatio_loss,label="val error",color="red")
plt.legend()
plt.savefig("valcurve.png")
plt.show()
print(f"last training value {train_loss_values[-1]}")

test(net,criterion,test_loader)