from tqdm import tqdm
from mnist_model import Mnist_Classifier, Mnist_Classifier_v2

import matplotlib.pyplot as plt
import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import os
import torch.optim as optim


def train(epoch, model, data_loader, phase = "training"):
  if phase == "training":
    model.train()
  if phase == "validation":
    model.eval()

  running_loss = 0.0
  running_correct = 0

  os.environ["CUDA_VISIBLE_DEVICES"] = "0"
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  # print(f"device for train : {torch.cuda.get_device_name()}")

  for batch_idx, (data, label) in enumerate(data_loader):
    if device == "cuda":
      data, label = data.to(device), label.to(device)
      model = model.to(device)

    if phase == "training":
      optimizer.zero_grad()

    output = model(data)
    # print(output.shape)
    # print(f"output after model : {output}\n")
    loss_function = nn.CrossEntropyLoss()
    loss = loss_function(output, label)
    # print(f"loss after CSE : {loss}\n")

    running_loss += loss
    # https://pytorch.org/docs/stable/generated/torch.max.html
    preds = output.detach().max(dim=1, keepdim=True)[1]
    # print(f"preds is {preds}")
    # https://discuss.pytorch.org/t/the-difference-between-data-and-detach/30926
    running_correct += preds.eq(label.data.view_as(preds)).cpu().sum()
    if phase == "training":
      loss.backward()
      optimizer.step()

  loss = running_loss / len(data_loader.dataset)
  accuracy = 100 * running_correct/len(data_loader.dataset)

  print(f'{phase} loss is {loss:{5}.{2}} and {phase} accuracy is {running_correct}/{len(data_loader.dataset)}{accuracy:{10}.{4}}')

  return loss, accuracy

if __name__ == "__main__":
    model = Mnist_Classifier_v2()
    model.load_state_dict(torch.load("./checkpoint36.pt")["model_state_dict"])

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    train_losses, train_accuracy = [], []
    val_losses, val_accuracy = [], []
    for epoch in tqdm(range(1, 31)):
    epoch_loss, epoch_accuracy = train(epoch, model, train_loader, phase='training')
    val_epoch_loss, val_epoch_accuracy = train(epoch, model, test_loader, phase='validation')
    train_losses.append(epoch_loss)
    train_accuracy.append(epoch_accuracy)
    val_losses.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)

    torch.save({
        'epoch': 30,
        'model_state_dict': model.state_dict(),
    }, './runs/v2checkpoint30.pt')

    train_losses = [tensor.detach() for tensor in train_losses]
    train_accuracy = [tensor.detach() for tensor in train_accuracy]
    val_losses = [tensor.detach() for tensor in val_losses]
    val_accuracy = [tensor.detach() for tensor in val_accuracy]
    plt.plot(range(1, len(train_losses) + 1), train_losses, 'bo', label='train loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, 'r', label = 'valid loss')
    plt.legend()
    plt.show()

    plt.plot(range(1, len(train_accuracy) + 1), train_accuracy, 'go', label='train accuracy')
    plt.plot(range(1, len(val_accuracy) + 1), val_accuracy, 'r', label = 'valid accuracy')
    plt.legend()
    plt.show()
