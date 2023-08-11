from PIL import Image
from mnist_model import Mnist_Classifier, Mnist_Classifier_v2
from mnist_loader import plot_img

import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt



if __name__ == "__main__":

    model = Mnist_Classifier_v2()
    model.load_state_dict(torch.load("./checkpoint36.pt")["model_state_dict"])
    model.eval()

    pics = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

    for i in pics:
        image = Image.open(f"./{i}.png").convert('L')
        inverted_image = Image.eval(image, lambda px: 255 - px)
        image = inverted_image.resize((28, 28))
        transformation = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,))])

        image = transformation(image)
        plot_img(image)
        image = image.unsqueeze(0)

        output = model(image)
        print(output)
        preds = output.detach().max(dim=1, keepdim=True)[1]
        print(preds)