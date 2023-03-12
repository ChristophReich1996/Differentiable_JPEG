import torch.nn as nn
from torch import Tensor

from .encode import jpeg_encode
from .decode import jpeg_decode


def jpeg_coding(image_rgb: Tensor, compression_strength: Tensor, differentiable: bool = True) -> Tensor:
    """Performs JPEG encoding.

    Args:
        image_rgb (Tensor): RGB input images of the shape [B, 3, H, W].
        compression_strength (Tensor): Compression strength of the shape [B].
        differentiable (bool): If true differentiable rounding is used. Default True.

    Returns:
        image_rgb_jpeg (Tensor): JPEG coded image of the shape [B, 3, H, W].
    """
    # Get original shape
    _, _, H, W = image_rgb.shape  # type: int, int, int, int
    # Perform encoding
    y_encoded, cb_encoded, cr_encoded = jpeg_encode(
        image_rgb=image_rgb, compression_strength=compression_strength, differentiable=differentiable
    )  # type: Tensor, Tensor, Tensor
    image_rgb_jpeg: Tensor = jpeg_decode(
        input_y=y_encoded, input_cb=cb_encoded, input_cr=cr_encoded, compression_strength=compression_strength, H=H, W=W
    )
    return image_rgb_jpeg


class JPEGCoding(nn.Module):
    """This class implements JPEG coding."""

    def __init__(self, differentiable: bool = True) -> None:
        """Constructor method.

        Args:
            differentiable (bool): If true differentiable rounding is used. Default True.
        """
        # Call super constructor
        super(JPEGCoding, self).__init__()
        # Save parameter
        self.differentiable: bool = differentiable

    def forward(self, image_rgb: Tensor, compression_strength: Tensor) -> Tensor:
        """Forward pass performs JPEG coding.

        Args:
            image_rgb (Tensor): RGB input images of the shape [B, 3, H, W].
            compression_strength (Tensor): Compression strength of the shape [B].

        Returns:
            image_rgb_jpeg (Tensor): JPEG coded image of the shape [B, 3, H, W].
        """
        # Perform coding
        image_rgb_jpeg: Tensor = jpeg_coding(
            image_rgb=image_rgb, compression_strength=compression_strength, differentiable=self.differentiable
        )
        return image_rgb_jpeg