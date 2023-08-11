import torch
import torch.nn as nn
import torch.functional as F

class Mnist_Classifier(nn.Module):
  def __init__(self):
    super().__init__()
    #Mnist Image (1 * 28 * 28)
    self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
    self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
    self.maxpool1 = nn.MaxPool2d(2)
    self.maxpool2 = nn.MaxPool2d(2)
    self.conv2_drop = nn.Dropout2d(0.3)
    self.lin_drop = nn.Dropout1d(0.3)
    self.fc1 = nn.Linear(320, 60)
    self.fc2 = nn.Linear(60, 10)

  def forward(self, x):
    x = F.relu(self.maxpool1(self.conv1(x)))
    x = F.relu(self.maxpool2(self.conv2(x)))
    x = x.view(-1, 320)
    x = F.relu(self.fc1(x))
    x = self.lin_drop(x)
    x = self.fc2(x)
    return F.softmax(x, dim=1)


class Mnist_Classifier_v2(nn.Module):
  def __init__(self):
    super().__init__()
    #Mnist Image (1 * 28 * 28)
    self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
    self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
    self.conv3 = nn.Conv2d(20, 40, kernel_size=3, padding=1)
    self.maxpool1 = nn.MaxPool2d(2)
    self.maxpool2 = nn.MaxPool2d(2)
    self.maxpool3 = nn.MaxPool2d(2)
    self.batchnorm1 = nn.BatchNorm2d(10)
    self.batchnorm2 = nn.BatchNorm2d(20)
    self.batchnorm3 = nn.BatchNorm2d(40)
    self.conv2_drop = nn.Dropout2d(0.3)
    self.lin_drop = nn.Dropout1d(0.3)
    self.fc1 = nn.Linear(160, 100)
    self.fc2 = nn.Linear(100, 40)
    self.fc3 = nn.Linear(40, 10)

  def forward(self, x):
    x = self.maxpool1(F.relu(self.batchnorm1(self.conv1(x))))
    x = self.maxpool2(F.relu(self.batchnorm2(self.conv2(x))))
    x = self.maxpool3(F.relu(self.batchnorm3(self.conv3(x))))
    x = x.view(-1, 160)
    x = F.relu(self.fc1(x))
    x = self.lin_drop(x)
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return F.softmax(x, dim=1)

