import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt

transformation = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = datasets.MNIST('data/', train=True, transform=transformation, download=True)
test_dataset = datasets.MNIST('data/', train=False, transform=transformation, download=True)

def load_mnist():
    transformation = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.1307,), (0.3081,))])

    train_dataset = datasets.MNIST('data/', train=True, transform=transformation, download=True)
    test_dataset = datasets.MNIST('data/', train=False, transform=transformation, download=True)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    return train_loader, test_loader

def plot_img(image):
    print(image.shape)
    image = image.numpy()[0]
    mean = 0.1307
    std = 0.3081
    image = ((mean * image) + std)
    plt.imshow(image, cmap = 'gray')
    plt.show()


if __name__ == "__main__":
    transformation = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST('data/', train=True, transform=transformation, download=True)
    test_dataset = datasets.MNIST('data/', train=False, transform=transformation, download=True)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    sample_data = next(iter(train_loader))
    print(sample_data[0].shape)
    print(len(sample_data))
    sample_test_data = iter(test_loader)
    plot_img(sample_data[0][1])
    print(sample_data[1][1])
    plot_img(sample_data[0][2])
    print(sample_data[1][2])