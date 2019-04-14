ABOUT = """Training script for CIFAR10, CIFAR100 and FashionMNIST"""

import yaml
import argparse
import torch as t
import numpy as np
import torchvision.transforms as T
from torchvision.datasets import CIFAR10, CIFAR100, FashionMNIST

from mobilenet import MobileNet

def accuracy(pred, target):
    """Computes accuracy from pred and given target classes"""
    return (pred.max(1)[1] == target).sum().float() / pred.size(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=ABOUT)
    parser.add_argument("--config", help="Path to yaml config", default="config.yml")
    args = parser.parse_args()
    config = yaml.load(open(args.config))

    # computing current dataset statistics on 100 randomly chosen samples
    train_ds = CIFAR10("~/workspace/data/cifar10", train=True, download=True)
    rand_indices = np.random.randint(len(train_ds), size=100)
    ds_mean = np.mean(np.concatenate([np.array(train_ds[i][0]) for i in rand_indices]), 1).mean(0)
    ds_std = np.std(np.concatenate([np.array(train_ds[i][0]) for i in rand_indices]), 1).mean(0)

    # making torch datasets using as transform normalization from previous step
    train_loader = t.utils.data.DataLoader(
        CIFAR10("~/workspace/data/cifar10", train=True, 
                transform=T.Compose([T.ToTensor(), T.Normalize(ds_mean, ds_std)])), 
        batch_size=config['batch_size'], shuffle=True)
    valid_loader = t.utils.data.DataLoader(
        CIFAR10("~/workspace/data/cifar10", train=False, 
                transform=T.Compose([T.ToTensor(), T.Normalize(ds_mean, ds_std)])), 
        batch_size=config['batch_size'])

    # defining model
    model = MobileNet(n_classes=len(train_ds.classes))
    model = model.to(config['device'])

    # defining loss criterion and optimizer
    criterion = t.nn.CrossEntropyLoss()
    opt = t.optim.Adam(model.parameters(), config['lr'])

    # training process
    for epoch in range(config['num_epochs']):
        # on training dataset
        model.train()
        for batch_idx, (inp, target) in enumerate(train_loader):
            inp, target = inp.to(config['device']), target.to(config['device'])
            opt.zero_grad()
            pred = model(inp)
            loss = criterion(pred, target)
            loss.backward()
            opt.step()
            if batch_idx % config['step_log'] == 0:
                print(f"Step {batch_idx}, loss {loss.item():.4f}, acc {accuracy(pred, target):.2%}")
        # validating after every epoch, printing to stdout
        val_acc = val_loss = n = 0
        model.eval()
        for inp, target in valid_loader:
            with t.no_grad():
                inp, target = inp.to(config['device']), target.to(config['device'])
                pred = model(inp)
                loss = criterion(pred, target)
                val_loss += loss.item()
                val_acc += accuracy(pred, target)
                n += 1
        print(f"Epoch {epoch + 1}, avg val loss {val_loss / n:.4f}, avg val acc {val_acc / n:.2%}")
    
