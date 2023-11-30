import numpy as np
from torch.utils.data import Dataset
from astropy.convolution import convolve
import torch
import pandas as pd

class Interpolator():
    def __init__(self):
        x = np.arange(0, 448)
        y = np.arange(0, 448)
        self.xx, self.yy = np.meshgrid(x, y)
        self.kernel = self.gaussian_kernel(19, 20)

    def gaussian_kernel(self, kernel_size, sigma):
        kernel = np.fromfunction(lambda x, y: (1 / (2 * np.pi * sigma**2)) * np.exp(-(
            (x - kernel_size//2)**2 + (y - kernel_size//2)**2) / (2 * sigma**2)), (kernel_size, kernel_size))
        normal = kernel / np.sum(kernel)
        return normal

    def __call__(self, arr):
        d = arr.diagonal().reshape((arr.shape[0], 1))
        mask = (d == 0)@((d == 0).T)
        arr = np.where(mask, np.nan, arr)
        newarr = convolve(arr, self.kernel)
        arr[mask == True] = newarr[mask == True]
        return np.nan_to_num(arr, nan=0.0)

class TorchDataset(Dataset):
    def __init__(self, file, fasta, cool, use_chip, root_dir="./", transform=None):
        """
        Arguments:
            file (string): Path to the file with annotations.
            root_dir (string): Directory with all the tfr files.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        # to store dictionary for genomic data
        self.df_frame = pd.read_csv(file)
        self.transform = transform
        self.fasta = fasta
        self.cooler = cool
        self.use_chip = use_chip
        self.i = Interpolator()

    def flatten(self, mat):
        return torch.tensor(mat[np.triu_indices(mat.shape[0], 2)]).unsqueeze(-1)

    def __len__(self):
        return len(self.df_frame)

    def __getitem__(self, idx):
        curr = self.df_frame.iloc[idx, :]
        # get monomer sequence
        X = self.fasta.fetch(curr["name"], curr["start"], curr["end"])
        # one-hot encoding
        word_to_ix = {"A": 0, "T": 1, "G": 2, "C": 3, "N": 4}
        X = [word_to_ix[w] for w in X]
        X = torch.eye(5)[X, :]
        X = X.T
        # apply transforms
        if self.transform:
            X = self.transform(X)
        # get HiC
        y = self.flatten(self.i(self.cooler.matrix(balance=False).fetch(
            (curr["name"], curr["start"], curr["end"])))[32:-32, 32:-32])

        if self.use_chip:
            c = curr["chip"].split(",")
            c = np.array([[int(j) for j in c]])
            return (X, c, y)

        return (X, y)
    
class RandomComplement():
    def __init__(self, dna_pair_columns, random_seed, transform_prob):
        """e.g. pairs = ((0, 1), (2, 3))"""
        self.transform_prob = transform_prob
        self.dna_pair_columns = dna_pair_columns
        self.random_generator = np.random.default_rng(random_seed)

    def __call__(self, x: torch.Tensor):
        if self.random_generator.random() > self.transform_prob:
            return x

        complement = torch.zeros_like(x)
        for pair in self.dna_pair_columns:
            complement[pair[0], :] = x[pair[1], :]
            complement[pair[1], :] = x[pair[0], :]

        complement[4, :] = x[4 :]

        return complement