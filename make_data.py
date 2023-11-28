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
import pyranges as pr

def process_bed(be,chr,start,end,window=800):
    df=be[chr,start:end+1].df
    curr=0
    ans=[]
    for i in range(start,end+1,window):
        if curr>=len(df): 
            ans.append(str("0"))
        elif int(df.iloc[curr]["Start"])>i: 
            ans.append(str("0"))
        elif int(df.iloc[curr]["Start"])<=i and i+window<int(df.iloc[curr]["End"]):
            ans.append(str(df.iloc[curr]["Name"]))
        elif int(df.iloc[curr]["Start"])<=i and i+window>=int(df.iloc[curr]["End"]):
            ans.append(str(df.iloc[curr]["Name"]))
            curr+=1
    return ans
def make_csv_file(clr,ref,window=512,occupancy_p=.6,filename="./data/hic_sequences.csv",be=None):
    fields=["name","start","end"]
    if be:
        fields=["name","start","end","chip"]
    rows=[]
    for chrome_name,chrome_size in zip(clr.chromnames,list(clr.chromsizes)):
        for i in range(0,chrome_size-window*int(clr.binsize),window*int(clr.binsize)):
            try :
                ref.fetch(chrome_name,i,min(i+window*clr.binsize,chrome_size-1))
                mat=clr.matrix(balance=False).fetch((chrome_name,i,min(i+window*clr.binsize,chrome_size-1)))
                diagonal=np.diag(mat)
                percentage_filled=np.sum((diagonal!=0))/diagonal.shape[0]
                if percentage_filled>occupancy_p:
                    if be:
                        c=process_bed(be,chrome_name,i,min(i+window*clr.binsize,chrome_size-1))
                        print(len(c))
                        rows.append([chrome_name,i,min(i+window*clr.binsize,chrome_size-1),",".join(c)])
                    else:
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
be=pr.read_bed('./data/bed/hglft_genome_13ec2_cf6d20.bed')

make_csv_file(ctrl,ref,filename="./data/hic_sequences_chip.csv",be=be)

if not os.path.isdir("./data/cvsDatasets"):
    os.makedirs("./data/cvsDatasets")
def make_batches(filename="./data/hic_sequences_chip.csv",loc="./data/cvsDatasets"):
    df=pd.read_csv(filename)
    df = df.sample(frac = 1)
    trainlen=int(.85*len(df))
    vallen=int(.1*len(df))
    train=df.iloc[:trainlen,:]
    val=df.iloc[trainlen:trainlen+vallen,:]
    test=df.iloc[trainlen+vallen:,:]
    train.to_csv(f'{loc}/train_chip.csv',index=False)
    test.to_csv(f'{loc}/test_chip.csv',index=False)
    val.to_csv(f'{loc}/val_chip.csv',index=False)

make_batches()