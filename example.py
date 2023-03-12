import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from torch import Tensor

from diff_jpeg import jpeg_encode, jpeg_decode, jpeg_coding, JPEGCoding


def coding() -> None:
    # Load test image and reshape to [B, 3, H, W]
    image: Tensor = torchvision.io.read_image("test_images/00000000.png").float()[None]
    # Init compression
    compression_strength: Tensor = torch.tensor([70], requires_grad=True, dtype=torch.float)
    # Perform coding
    image_jpeg: Tensor = jpeg_coding(image_rgb=image, compression_strength=compression_strength, differentiable=True)
    # Print L1 distance
    print("L1 distance is:", (image_jpeg - image).abs().mean().item())
    # Plot original image and coded image
    plt.imshow(image[0].permute(1, 2, 0).detach() / 255.0)
    plt.show()
    plt.imshow(image_jpeg[0].permute(1, 2, 0).detach().clip(min=0, max=255) / 255.0)
    plt.show()


def coding_class() -> None:
    # Load test image and reshape to [B, 3, H, W]
    image: Tensor = torchvision.io.read_image("test_images/00000000.png").float()[None]
    # Init compression
    compression_strength: Tensor = torch.tensor([55], requires_grad=True, dtype=torch.float)
    # Init JPEG coding class
    jpeg_coding_module: nn.Module = JPEGCoding(differentiable=True)
    # Perform coding
    image_jpeg: Tensor = jpeg_coding_module(image_rgb=image, compression_strength=compression_strength)
    # Print L1 distance
    print("L1 distance is:", (image_jpeg - image).abs().mean().item())
    # Plot original image and coded image
    plt.imshow(image[0].permute(1, 2, 0).detach() / 255.0)
    plt.show()
    plt.imshow(image_jpeg[0].permute(1, 2, 0).detach().clip(min=0, max=255) / 255.0)
    plt.show()


def encode_decode() -> None:
    # Load test image and reshape to [B, 3, H, W]
    image: Tensor = torchvision.io.read_image("test_images/00000000.png").float()[None]
    # Init compression
    compression_strength: Tensor = torch.tensor([5], requires_grad=True, dtype=torch.float)
    # Perform encoding
    y_encoded, cb_encoded, cr_encoded = jpeg_encode(image, compression_strength, True)
    # Perform decoding
    image_jpeg: Tensor = jpeg_decode(y_encoded, cb_encoded, cr_encoded, compression_strength, 720, 1280)
    # Print L1 distance
    print("L1 distance is:", (image_jpeg - image).abs().mean().item())
    # Plot original image and coded image
    plt.imshow(image[0].permute(1, 2, 0).detach() / 255.0)
    plt.show()
    plt.imshow(image_jpeg[0].permute(1, 2, 0).detach().clip(min=0, max=255) / 255.0)
    plt.show()


def main() -> None:
    coding()
    coding_class()
    encode_decode()


if __name__ == "__main__":
    main()
