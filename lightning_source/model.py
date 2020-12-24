
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.optim as optim


class LitNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d()
        self.fc1 = nn.Linear(9216, 128)
        self.dropout2 = nn.Dropout2d()
        self.fc2 = nn.Linear(128, 10)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = x.view(-1, 9216)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)

        x = F.log_softmax(x, dim=1)
        return x

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        # forward + backward + optimize
        logits = self(inputs)
        loss = self.criterion(logits, labels)
        return loss

    def training_epoch_end(self, losses):
        print(np.sum(losses))
        return np.sum(losses)

    def test_step():
        inputs, labels = batch
        # forward + backward + optimize
        logits = self(inputs)
        loss = self.criterion(logits, labels)

        _, predicted = torch.max(logits, 1)
        acc = (predicted == labels).sum().item()

        return {"loss": loss, "acc": acc}

    def test_epoch_end(self, test_results):
        acc = 0
        for results in test_results:
            acc += test_results["acc"]
        print(acc)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(),
                          lr=1e-3, momentum=0.99, nesterov=True)

