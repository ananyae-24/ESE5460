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
    for chrome_name,chrome_size in zip(clr.chromnames,list(clr.chromsizes)):
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

if not os.path.isdir("./data/cvsDatasets"):
    os.makedirs("./data/cvsDatasets")
def make_batches(filename="./data/hic_sequences.csv",loc="./data/cvsDatasets"):
    df=pd.read_csv(filename)
    df = df.sample(frac = 1)
    trainlen=int(.95*len(df))

    train=df.iloc[:trainlen,:]
    test=df.iloc[trainlen:,:]
    train.to_csv(f'{loc}/train.csv',index=False)
    train.to_csv(f'{loc}/test.csv',index=False)

make_batches()