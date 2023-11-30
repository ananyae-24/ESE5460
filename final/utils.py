import torch
import matplotlib as plt

def plot_hic(x, name="hello"):
    mat = torch.zeros(448, 448)
    idx = torch.triu_indices(448, 448, 2)
    mat[idx[0], idx[1]] = x.float()
    im = plt.imshow(mat+mat.T, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04, label='balanced')
    plt.savefig(f"{name}.png")
    plt.show()