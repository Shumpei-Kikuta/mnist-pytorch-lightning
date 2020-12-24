import argparse
import time
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

from model import LitNet
from dataset import MNISTDataModule


def parser():
    '''
    argument
    '''
    parser = argparse.ArgumentParser(description='PyTorch MNIST')
    parser.add_argument('--epochs', '-e', type=int, default=2,
                        help='number of epochs to train (default: 2)')
    parser.add_argument('--lr', '-l', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    args = parser.parse_args()
    return args


def main():
    '''
    pytorch version
    1. 見通しが悪い: エンジニアリングコードとDLコードが混ざっていて、可読性が低くなりがち
        args, train_loop, modelの設定など基本的にmain関数での実行が必要
    2. エラーが起こりやすい: 必要なステップが多いため、コードミスが起こりやすい
        optimizer.step, loss.backwardなど忘れやすい
    3. 保守性が低い: コードの凝集性が低く、手続き的なコードになりがち
        data周りの設定(dataset, loader), optimizerの設定、modelの設定などが広くmainに書かれるため、
        一箇所の影響が広く及び、変更しづらい。

    Pytorch-lightningなら、
    1. 可読性が上がる: エンジニアリングコードとDLコードを完全に分離できる
    2. エラーが起こりづらい: 必要な関数がないとそもそも動かない
    3. 保守性が上がる: よりOOPで書けるため、役割ごとにコードをまとめられる(凝集性上がる)
    '''
    args = parser()

    net = LitNet()
    trainer = pl.Trainer()
    dm = MNISTDataModule()

    trainer.fit(net, dm)

    # test関数
    trainer.test(model=net, test_dataloaders=dm.test_dataloader)


if __name__ == '__main__':
    start_time = time.time()
    main()
    print('elapsed time: {:.3f} [sec]'.format(time.time() - start_time))