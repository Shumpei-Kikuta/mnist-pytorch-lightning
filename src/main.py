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

from model import Net


def parser():
    '''
    argument
    '''
    parser = argparse.ArgumentParser(description='PyTorch MNIST')
    parser.add_argument('--epochs', '-e', type=int, default=2,
                        help='number of epochs to train (default: 2)')
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

    STEP
    1. dataset moduleの定義
    2. Networkの定義(configure_optimizers, train_step)

    '''
    args = parser()

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, ), (0.5, ))])

    # 1. datasetの定義
    trainset = MNIST(root='./data',
                     train=True,
                     download=True,
                     transform=transform)
    testset = MNIST(root='./data',
                    train=False,
                    download=True,
                    transform=transform)

    # 2. loaderの定義
    trainloader = DataLoader(trainset,
                             batch_size=100,
                             shuffle=True,
                             num_workers=2)
    testloader = DataLoader(testset,
                            batch_size=100,
                            shuffle=False,
                            num_workers=2)

    # 3. modelの定義
    net = Net()

    # 4. loss・optimizerの定義
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(),
                          lr=1e-2, momentum=0.99, nesterov=True)

    # 5. train関数
    for epoch in range(args.epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader, 0):
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:
                print('[{:d}, {:5d}] loss: {:.3f}'
                      .format(epoch+1, i+1, running_loss/100))
                running_loss = 0.0
    print('Finished Training')

    # test関数
    correct = 0
    total = 0
    with torch.no_grad():
        for (images, labels) in testloader:
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy: {:.2f} %'.format(100 * float(correct/total)))


if __name__ == '__main__':
    start_time = time.time()
    main()
    print('elapsed time: {:.3f} [sec]'.format(time.time() - start_time))