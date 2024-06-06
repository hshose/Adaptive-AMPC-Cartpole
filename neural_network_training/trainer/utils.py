import os
import random
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch import nn
from tqdm import tqdm
import sys
from pathlib import Path

from torch.utils import data


# Adapted from https://github.com/pytorch/tnt/blob/master/torchnet/meter/averagevaluemeter.py (broken url: 10/2022)
# new source https://tnt.readthedocs.io/en/latest/_modules/torchnet/meter/averagevaluemeter.html

# TODO: add comments
def create_dir_if_not_exists(dir: Path):
    # if not (dir.exists() and dir.is_dir()):
    Path.mkdir(dir, parents=True, exist_ok=True)


def create_empty_file(file_path: Path):
    create_dir_if_not_exists(file_path.parent)
    Path(file_path).touch()


def block_printing(func):
    """Function decorator that blocks output from the print function to the console."""
    def func_wrapper(*args, **kwargs):
        # block all printing to the console
        sys.stdout = open(os.devnull, 'w')
        # call the method in question
        value = func(*args, **kwargs)
        # enable all printing to the console
        sys.stdout = sys.__stdout__
        # pass the return value of the method back
        return value

    return func_wrapper


class AverageMeter:
    """Class for metering an arithmetic average value."""

    def __init__(self):
        self.__n = 0
        self.__sum = 0.0
        self.__mean = 0.0
        self.history = []

    def add(self, value):
        self.__sum += value
        self.__n += 1
        self.__mean = self.__sum / self.__n

        self.history.append(self.value())

    def value(self):
        return self.__mean

    def reset(self):
        self.__n = 0
        self.__sum = 0.0
        self.__mean = 0.0
        self.history = []

    def plot(self, fname, xlabel=None, ylabel=None):
        lineplot(
            x=np.arange(len(self.history)),
            y=[x.item() for x in self.history],
            fname=fname,
            xlabel=xlabel, ylabel=ylabel)


class ExponentialAverageMeter:
    """Class for metering an exponential average value."""

    def __init__(self, gamma: float = .98):
        assert 0 <= gamma < 1
        self.__n = 0
        self.__mean = 0.0
        self.history = []
        self.gamma = gamma

    def add(self, value):
        self.__mean = self.__mean * self.gamma + (1 - self.gamma) * value
        self.__n += 1

        self.history.append(self.value())

    def value(self):
        if self.__n == 0:
            return 0
        return self.__mean / (1 - self.gamma ** self.__n)

    def reset(self):
        self.__n = 0
        self.__mean = 0
        self.history = []

    def plot(self, fname, xlabel=None, ylabel=None):
        lineplot(
            x=np.arange(len(self.history)),
            y=[x.item() for x in self.history],
            fname=fname,
            xlabel=xlabel, ylabel=ylabel)

    def to_csv(self, fname, xlabel, ylabel):
        to_csv(
            x=np.arange(len(self.history)),
            y=[x.item() for x in self.history],
            fname=fname,
            xlabel=xlabel, ylabel=ylabel)


class Metrics:
    """Metrics class used to represent trainer metrics."""

    def __init__(self, gamma: float = .98):
        assert 0 <= gamma <= 1

        self.gamma = gamma
        self.averages = dict()

    def add(self, name: str, value: float):
        if name not in self.averages:
            self.averages[name] = ExponentialAverageMeter(self.gamma)
        self.averages[name].add(value)

    def get_value(self, name):
        assert name in self.averages
        return self.averages[name].value()

    def get_history(self, name):
        assert name in self.averages
        return self.averages[name].history

    def reset(self):
        for avg in self.averages.values():
            avg.reset()

    def plot(self, path_prefix: Path):
        for name, avg in self.averages.items():
            avg.plot(path_prefix / f"{name}.png", 'steps', name)

    def to_csv(self, path_prefix: Path):
        for name, avg in self.averages.items():
            avg.to_csv(path_prefix / f"{name}.csv", 'steps', name)


def lineplot(x, y, fname: Path, xlabel=None, ylabel=None):
    """Draws and saves a lineplot."""
    if not os.path.exists(os.path.dirname(fname)):
        os.makedirs(os.path.dirname(fname))

    plot = sns.lineplot(x=x, y=y)
    fig = plot.get_figure()

    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    fig.savefig(fname)

    plt.clf()


def to_csv(x, y, fname: Path, xlabel, ylabel):
    """Saves a plot to csv."""
    if not os.path.exists(os.path.dirname(fname)):
        os.makedirs(os.path.dirname(fname))

    df = pd.DataFrame({
        xlabel: x,
        ylabel: y
    })
    df.to_csv(fname, index=False)


def show_batch(
        imgs: torch.tensor,
        height: int,
        width: int,
        figsize: Tuple[float] = (10, 10)
):
    """Show batch of images"""

    assert len(imgs.shape) == 4
    assert height * width <= imgs.shape[0]

    # Turn normalized images back to values between 0 and 1
    mean_imagenet = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std_imagenet = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

    imgs = imgs * std_imagenet
    imgs = imgs + mean_imagenet

    # Make grid of images
    fig, axs = plt.subplots(height, width, figsize=figsize)
    fig.tight_layout()

    for y in range(height):
        for x in range(width):
            axs[y][x].imshow(imgs[y + width * x].permute(1, 2, 0))
            axs[y][x].axis('off')


def to_onnx(model, path: Path, inp_shape):
    """Exports a model to the onnx file format."""
    model = torch.jit.script(model)
    torch.onnx.export(model, torch.zeros(inp_shape), path, verbose=False,
                      input_names=["input"], output_names=["output"])


if __name__ == '__main__':
    from collections import OrderedDict
    model = torch.nn.Sequential(OrderedDict([
        ('gemm1', torch.nn.Linear(1, 1)),
        ('relu1', torch.nn.ReLU()),
        ('gemm2', torch.nn.Linear(1, 1)),
    ]))
    torch.save(model, Path(f"C:/Users/Johannes/OneDrive - Students RWTH Aachen University/DSME/BA/node_dummy/tiny.pth"))
    to_onnx(model, Path(f"C:/Users/Johannes/OneDrive - Students RWTH Aachen University/DSME/BA/node_dummy/tiny.onnx"),
            (1, 1))


def accuracy(model, dataloader, device):
    """Returns the evaluation accuracy of 'model'."""
    model.eval()

    acc = 0

    with torch.no_grad():
        for batch in tqdm(dataloader):
            imgs, targets = batch

            imgs = imgs.to(device)
            targets = targets.to(device)

            outs = model(imgs)

            acc += (outs.detach().argmax(dim=-1) == targets).float().mean()

    acc = acc.cpu() / len(dataloader)

    return acc


def flatten(model):
    def model_to_list(model):
        children = list(model.children())

        if len(children) > 0:
            modules = []
            for child in children:
                modules += model_to_list(child)
            return modules
        return [model]

    return nn.Sequential(*model_to_list(model))


def n_elements(slices):
    """Returns the number of valid elements in 'slices'."""
    counter = 1

    for s in slices:
        if s.start is not None and s.stop is not None:
            if s.step is not None:
                counter *= np.ceil((s.stop - s.start) / s.step)
            else:
                counter *= (s.stop - s.start)

    return counter


def model_size(model):
    """Returns the model size stored on local disk."""
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p")
    os.remove('temp.p')

    return size / 1e3


def num_parameters(model):
    """Returns the number of model parameters."""
    return sum([torch.numel(param) for param in model.parameters()])


def set_seed(seed: int):
    """Sets seed for multiple libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def get_dataloader(
        ds,
        batch_size: int,
        num_workers: int,
        pin_memory: bool = True,  # accelerates copy operation to GPU
        **kwargs):
    """Shortcut to get the DataLoader of 'ds' with default settings from config."""
    return data.DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        # shuffle=True,
        **kwargs)
