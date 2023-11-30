import torchvision as thv
from torch.utils.data import DataLoader
import cooler
import pysam
import torch
import matplotlib.pyplot as plt

import model
import data
import training

def main():
    RANDOM_SEED = 42
    BATCH_SIZE = 2
    NUM_EPOCHS = 30
    DATA_PATH = '../data'
    USE_CHIP_SEQ = False
    USE_AUTOCAST = False

    hic_file = cooler.Cooler(f'{DATA_PATH}/coolers/H1hESC_hg38_4DNFI1O6IL1Q.mapq_30.2048.cool')
    genome_file = pysam.FastaFile(f'{DATA_PATH}/hg38.ml.fa')

    transform = thv.transforms.Compose((
        data.RandomComplement(((0, 1), (2, 3)), RANDOM_SEED, 0.5)
    ))
    train_set = data.TorchDataset(
        f'{DATA_PATH}/cvsDatasets/train_chip.csv',
        genome_file,
        hic_file,
        USE_CHIP_SEQ,
        transform=transform
    )
    val_set = data.TorchDataset(
        f'{DATA_PATH}/cvsDatasets/val_chip.csv',
        genome_file,
        hic_file,
        USE_CHIP_SEQ
    )
    test_set = data.TorchDataset(
        f'{DATA_PATH}/data/cvsDatasets/test_chip.csv',
        genome_file,
        hic_file,
        USE_CHIP_SEQ
    )
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device}")

    if USE_CHIP_SEQ:
        net = model.ChipSeqModel(5)
    else:
        net = model.BaseModel(5)
    
    criterion = training.WeightedMSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001)
    train_loss_values, validation_loss = training.train(
        net,
        optimizer,
        criterion,
        train_loader,
        val_loader,
        NUM_EPOCHS,
        USE_AUTOCAST,
        model_name="Akita"
    )
    torch.save(net.state_dict(), "./model.pth")

    plt.plot(list(range(len(train_loss_values))),
            train_loss_values, label="training error", color="red")
    plt.legend()
    plt.savefig("trainingcurve.png")
    plt.show()

    plt.plot(list(range(len(validation_loss))),
            validation_loss, label="val error", color="red")
    plt.legend()
    plt.savefig("valcurve.png")
    plt.show()
    print(f"last training value {train_loss_values[-1]}")

    training.test(
        net,
        criterion, 
        test_loader,
        USE_AUTOCAST
    )


if __name__ == '__main__':
    main()