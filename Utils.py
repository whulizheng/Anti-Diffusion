from torchvision.io import read_image
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import torch

import os

def torch_to_pill(images):
    if images.ndim == 3:
        images = images[None, ...]
    images = images.detach()
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.cpu().permute(0, 2, 3, 1).float().numpy()
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        pil_images = [Image.fromarray(image.squeeze(), mode="L")
                      for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]
    return pil_images

def load_image(image_path, image_size, device=None):
    image = read_image(image_path)
    image = image[:3].unsqueeze_(0).float() / 127.5 - 1.  # [-1, 1]
    image = F.interpolate(image, (image_size, image_size))
    if device is not None:
        image = image.to(device)
    return image[0]


class ImageData(Dataset):
    def __init__(self, path, image_size, transform=None):
        self.imgType_list = ['jpg', 'png', 'jpeg']
        self.image_size = image_size
        self.path = path
        self.images = []
        self.transform = transform
        for filepath, dirnames, filenames in os.walk(self.path):
            for filename in filenames:
                if filename.split(".")[-1] in self.imgType_list:
                    self.images.append(os.path.join(filepath, filename))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        file_name = image.split("/")[-1]
        image = load_image(image, self.image_size)
        return file_name, image


def save_images(images, image_names, args):
    images = torch_to_pill(images)
    for i, image in enumerate(images):
        image.save(os.path.join(
            args.save_dir, image_names[i]))