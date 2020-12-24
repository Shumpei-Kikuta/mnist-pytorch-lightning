import argparse
import time
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import torch.optim as optim
import pytorch_lightning as pl


class MNISTDataModule(pl.LightningDataModule):

      def __init__(self, batch_size=32):
          super().__init__()
          self.batch_size = batch_size

      # OPTIONAL, called only on 1 GPU/machine
      # TODO: 度のタイミングで読み込まれるか
      def prepare_data(self):
          MNIST(os.getcwd(), train=True, download=True)
          MNIST(os.getcwd(), train=False, download=True)

      # OPTIONAL, called for every GPU/machine (assigning state is OK)
      # TODO: 度のタイミングで読み込まれるか
      def setup(self, stage):
          # transforms
          transform=transforms.Compose([
              transforms.ToTensor(),
              transforms.Normalize((0.1307,), (0.3081,))
          ])
          # split dataset
          self.mnist_train = MNIST(os.getcwd(), train=True, transform=transform)
          self.mnist_test = MNIST(os.getcwd(), train=False, transform=transform)

      # return the dataloader for each split
      def train_dataloader(self):
          mnist_train = DataLoader(self.mnist_train, batch_size=self.batch_size)
          return mnist_train

      def test_dataloader(self):
          mnist_test = DataLoader(self.mnist_test, batch_size=self.batch_size)
          return mnist_test