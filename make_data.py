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


def make_csv_file(clr,ref,window=512,occupancy_p=.6,filename="./data/hic_sequences.csv"):
    fields=["name","start","end"]
    rows=[]
    for chrome_name,chrome_size in zip(clr.chromnames[:1],list(clr.chromsizes)[:1]):
        for i in range(0,chrome_size-window*clr.binsize,window*clr.binsize):
            try :
                ref.fetch(chrome_name,i,min(i+window*clr.binsize,chrome_size-1))
                mat=clr.matrix(balance=False).fetch((chrome_name,i,min(i+window*clr.binsize,chrome_size-1)))
                diagonal=np.diag(mat)
                percentage_filled=np.sum((diagonal!=0))/diagonal.shape[0]
                if percentage_filled>occupancy_p:
                    rows.append([chrome_name,i,min(i+window*clr.binsize,chrome_size-1)])
            except:
                print(f"chromosome not found {chrome_name}")

    with open(filename, 'w') as csvfile:  
        # creating a csv writer object  
        csvwriter = csv.writer(csvfile)  
        
        # writing the fields  
        csvwriter.writerow(fields)  
        
        # writing the data rows  
        csvwriter.writerows(rows) 

ctrl = cooler.Cooler('./data/coolers/H1hESC_hg38_4DNFI1O6IL1Q.mapq_30.2048.cool')
ref = pysam.FastaFile('./data/hg38.ml.fa')

make_csv_file(ctrl,ref)

class CreateData():
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
        self.data_set_X,self.data_set_Y=[],[]
        for index, row in self.df_frame.iterrows():
            curr=row
            X=self.one_hot_enc(self.fasta.fetch(curr["name"],curr["start"],curr["end"]))
            y=self.flatten(self.cooler.matrix(balance=False).fetch((curr["name"],curr["start"],curr["end"]))[32:-32,32:-32])
            self.data_set_X.append(X),self.data_set_Y.append(y)
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
    def __call__(self):
        return (self.data_set_X,self.data_set_Y)
class TorchDataset(Dataset):
    def __init__(self,data,transform=None):
        super().__init__()
        self.transform=transform
        self.data=data
    def __len__(self):
        return len(self.data[0])
    def __getitem__(self, idx):
        X,y=self.data[0][idx],self.data[1][idx]
        sample=(X,y)
        return sample
    
C=CreateData("./hic_sequences.csv",ref,ctrl)
dataset = TorchDataset(C())
train,test=torch.utils.data.random_split(dataset,[int(0.9*len(dataset)),len(dataset)-int(0.90*len(dataset))], generator=torch.Generator().manual_seed(42))

torch.save(train,"./data/train.pth")
torch.save(test,"./data/test.pth")
