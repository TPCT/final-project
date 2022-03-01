from torch.utils.data import DataLoader, random_split
from torch.nn import Sequential, ReLU, MaxPool2d, Conv2d, Flatten, Linear, Module
from torch.optim import Adam
from genericFunctions import GenericFunctions, DeviceDataLoader
from ImageClassificationBase import ImageClassificationBase
from random import random
import torch


class SimpleResidualBlock(Module):
    def __init__(self, input_channels=3, output_channels=3, kernel_size=3):
        super(SimpleResidualBlock, self).__init__()
        self.convolutional_layer1 = Conv2d(input_channels, output_channels, kernel_size, padding=1)
        self.activation_function1 = ReLU()
        self.convolutional_layer2 = Conv2d(input_channels, output_channels, kernel_size, padding=1)
        self.activation_function2 = ReLU()

    def forward(self, batch):
        output = self.convolutional_layer1(batch)
        output = self.activation_function1(output)
        output = self.convolutional_layer2(batch)
        output = self.activation_function2(output) + batch
        return output
