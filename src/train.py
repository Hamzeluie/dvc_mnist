import os
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).parents[1].as_posix()
sys.path.append(ROOT_DIR)

from torchvision import transforms
from torchvision.datasets.mnist import MNIST
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
import pandas as pd
import numpy as np
from src.modules import TrainBaseModule
from tqdm import tqdm
from dvclive import Live
import yaml


device = "cuda"
# dvc exp run -S "mnist.lr=0.1" -S "mnist.momentum=2"

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # The first argument specifies the number of input channels (1 for grayscale images), 
        # the second argument specifies the number of output channels (32 and 64 respectively), 
        # and the third argument specifies the kernel size (3x3 convolutional kernels).
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 10)

# the forward method defines the forward pass of the neural network, specifying how input data flows through the layers.
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 12 * 12)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DvcMnistTrain(TrainBaseModule):
    def __init__(self, epochs, lr, momentum):
        super().__init__()
        self.epochs = epochs
        self.lr = lr
        self.momentum = momentum
        self.net = Net().to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr = self.lr, momentum = self.momentum)
        self.BATCH_SIZE = 4
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        ds = MNIST(os.path.join(ROOT_DIR, "data"), download=True, transform=transform)
        train_set, val_set = torch.utils.data.random_split(ds, [50000, 10000])
        self.train_loader = DataLoader(train_set, batch_size=self.BATCH_SIZE, shuffle=True)
        self.test_loader = DataLoader(val_set, batch_size=self.BATCH_SIZE, shuffle=True)
    
    def train(self):
        with Live(os.path.join(ROOT_DIR, "results")) as live:
            for epoch in range(self.epochs):
                print(f'Starting Epoch: {epoch + 1}...')
                loss = self.__train_one_epoch(epoch)
                acc = self.validation_evaluation()
                live.log_metric("loss", loss)
                live.log_metric("accuracy", acc)
                live.next_step()
    
    def __train_one_epoch(self, epoch):
        running_loss = 0.0
        with tqdm(total=self.train_loader.__len__() * self.BATCH_SIZE, desc=f'Epoch {epoch+1}/{self.epochs}', unit='img') as pbar:
            for i, data in enumerate(self.train_loader):
                #Training Loop
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                loss = self.__train_one_step(inputs, labels)
                running_loss += loss
                pbar.update(self.BATCH_SIZE)
                pbar.set_postfix(**{'loss': round(loss, 5)})
            return running_loss
    
    def __train_one_step(self, x, y):
        self.optimizer.zero_grad()
        outputs = self.net(x.float())
        loss = self.criterion(outputs,y )
        loss.backward()
        self.optimizer.step()
        return loss.item()
        
    def validation_evaluation(self):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.test_loader:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                outputs = self.net(images.float())

                _, predicted = torch.max(outputs.data, dim = 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
        return accuracy


if __name__ == '__main__':
    train_params = yaml.safe_load(open(os.path.join(ROOT_DIR, "params.yaml")))["mnist"]
    train_finger_gan = DvcMnistTrain(train_params["epochs"], train_params["lr"], train_params["momentum"])
    train_finger_gan.train()
