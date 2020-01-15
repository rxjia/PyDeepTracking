import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

loader = transforms.Compose([transforms.ToTensor()])
unloader = transforms.ToPILImage()


def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image


def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def save_tensor_img(tensor, file):
    with torch.no_grad():
        # img=tensor_to_PIL(tensor)
        npimg = (tensor.cpu().clone().squeeze(0).numpy() * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(npimg)
        img.save(file)
