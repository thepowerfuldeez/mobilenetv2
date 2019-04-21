ABOUT = """Training script for CIFAR10, CIFAR100 and FashionMNIST"""

import yaml
import json
import argparse
import torch as t
import numpy as np
import torchvision.transforms as T
import torchvision.models as tvm
from torchvision.datasets import CIFAR10, CIFAR100, FashionMNIST

from mobilenet import MobileNet

def SGDR(optimizer, cycle_len=3, cycle_mult=1.2):
    """
    SGD with restarts optimizer
    :param optimizer: torch optimizer instance
    :param cycle_len: length of one cycle (in epochs)
    :param cycle_mult: multiplier after one cycle passed
    """
    def schedule_func(epoch):
        if not hasattr(schedule_func, 'T_start'):
            schedule_func.T_start = 0
            schedule_func.T_length = cycle_len
            schedule_func.T_next = cycle_len
        if epoch == schedule_func.T_next:
            schedule_func.T_start = schedule_func.T_next
            schedule_func.T_length *= cycle_mult
            schedule_func.T_next += schedule_func.T_length
        T_cur = epoch - schedule_func.T_start
        lr_coef = 0.5 * (1 + np.cos(T_cur / schedule_func.T_length * np.pi))
        print(f"Epoch {epoch}, lr_coef == {lr_coef}")
        return lr_coef
    return t.optim.lr_scheduler.LambdaLR(optimizer, schedule_func)

def accuracy(pred, target):
    """Computes accuracy from pred and given target classes"""
    return (pred.max(1)[1] == target).sum().float() / pred.size(0)

def get_loaders(ds_name="cifar10"):
    """
    Return train and valid loaders for different datasets
    """
    if ds_name == "cifar10":
        ds_path = "~/workspace/data/cifar10"
        ds_class = CIFAR10
    elif ds_name == "cifar100":
        ds_path = "~/workspace/data/cifar100"
        ds_class = CIFAR100

    # making torch datasets using normalization and augmentations as transform
    train_tfm = T.Compose([T.RandomAffine(10), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    valid_tfm = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_loader = t.utils.data.DataLoader(
        ds_class(ds_path, train=True, 
                transform=train_tfm), 
        batch_size=config['batch_size'], shuffle=True)
    valid_loader = t.utils.data.DataLoader(
        ds_class(ds_path, train=False, 
                transform=valid_tfm), 
        batch_size=config['batch_size'])
    return train_loader, valid_loader

def train(config, model, train_loader, valid_loader, criterion, opt, sched):
    """
    Train procedure: iterate num_epochs times over train and valid loaders,
    return valid loss and valid accuracy 
    """
    tr_accuracy = []
    tr_loss = []
    valid_loss = []
    valid_accuracy = []
    for epoch in range(config['num_epochs']):
        model.train()
        for batch_idx, (inp, target) in enumerate(train_loader):
            inp, target = inp.to(config['device']), target.to(config['device'])
            opt.zero_grad()
            pred = model(inp)
            loss = criterion(pred, target)
            loss.backward()
            opt.step()
            if batch_idx % config['step_log'] == 0:
                loss_item = loss.item()
                tr_acc = accuracy(pred, target)
                print(f"Step {batch_idx}, loss {loss_item:.4f}, acc {tr_acc:.2%}")
                tr_loss.append(loss_item)
                tr_accuracy.append(tr_acc)
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
        print(pred.max(1)[1])
        val_loss /= n
        val_acc /= n
        valid_loss.append(val_loss)
        valid_accuracy.append(val_acc)
        print(f"Epoch {epoch + 1}, avg val loss {val_loss:.4f}, avg val acc {val_acc:.2%}")
        sched.step()
    return tr_loss, tr_accuracy, valid_loss, valid_accuracy

def run_training(config, n_classes, train_loader, valid_loader, width=1):
    """
    Whole training procedure with fine-tune after regular training
    """
    # defining model
    if width > 1:
        model = tvm.resnet18(num_classes=n_classes)
    else:
        model = MobileNet(n_classes=n_classes, width_mult=width)
    model = model.to(config['device'])

    # print out number of parameters
    num_params = 0
    for p in model.parameters():
        num_params += np.prod(p.size())
    print(f"width={width}, num_params {num_params}")

    # defining loss criterion, optimizer and learning rate scheduler
    criterion = t.nn.CrossEntropyLoss()
    opt = t.optim.Adam(model.parameters(), config['lr'])
    sched = t.optim.lr_scheduler.MultiStepLR(opt, [3, 6])
    
    # training process with Adam
    tr_loss, tr_accuracy, valid_loss, valid_accuracy = train(config, model, train_loader, valid_loader, criterion, opt, sched)
    # training process with SGDR
    opt = t.optim.SGD(model.parameters(), config['lr'] / 10, momentum=0.9)
    sched = SGDR(opt, 3, 1.2)
    tr_loss_finetune, tr_accuracy_finetune, valid_loss_finetune, valid_accuracy_finetune = train(config, model, train_loader, valid_loader, criterion, opt, sched)
    return [tr_loss + tr_loss_finetune, tr_accuracy + tr_accuracy_finetune, 
            valid_loss + valid_loss_finetune, valid_accuracy + valid_accuracy_finetune]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=ABOUT)
    parser.add_argument("--config", help="Path to yaml config", default="config.yml")
    args = parser.parse_args()
    config = yaml.load(open(args.config))
    config['lr'] = float(config['lr'])

    stats = {}
    for ds_name in ['cifar10', 'cifar100']:
        train_loader, valid_loader = get_loaders(ds_name)
        ds_stats = {}
        for width in [1.1, 1, 0.75, 0.5, 0.25, 0.032]:
            if width > 1:
                name = "resnet18"
            else:
                name = f"mobilenetv1_{width}"
            ds_stats[name] = run_training(config, len(train_loader.dataset.classes), train_loader, valid_loader, width)
        stats[ds_name] = ds_stats
    json.dump(stats, open("stats.json", "w"))
