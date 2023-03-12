from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

M: Tensor = torch.tensor(
    [
        [0.299, 0.587, 0.114],
        [-0.168736, -0.331264, 0.5],
        [0.5, -0.418688, -0.081312],
    ],
    dtype=torch.float,
)

B = torch.tensor([0, 128, 128], dtype=torch.float)


def rgb_to_ycbcr(input_rgb: Tensor) -> Tensor:
    """

    Args:
        input_rgb (Tensor): RGB input tensor of the shape [*, 3].

    Returns:
        output_ycbcr (Tensor): YCbCr output tensor of the shape [*, 3].
    """
    # Check if input is a tensor with the correct shape
    assert isinstance(input_rgb, Tensor), f"Given compression strength must be a torch.Tensor, got {type(input_rgb)}."
    assert input_rgb.shape[-1] == 3, f"Last axis of the input must have 3 dimensions, got {input_rgb.shape[-1]}."
    # Get original shape and dtype
    dtype: torch.dtype = input_rgb.dtype
    device: torch.device = input_rgb.device
    # Convert from RGB to YCbCr
    output_ycbcr: Tensor = torch.einsum("ij, ...j -> ...i", M.to(dtype=dtype, device=device), input_rgb)
    output_ycbcr = output_ycbcr + B.to(dtype=dtype, device=device)
    return output_ycbcr


def chroma_subsampling(input_ycbcr: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """This function implements chroma subsampling as an avg. pool operation.

    Args:
        input_ycbcr (Tensor): YCbCr input tensor of the shape [B, 3, H, W].

    Returns:
        output_y (Tensor): Y component (not-subsampled), shape is [B, H, W].
        output_cb (Tensor): Cb component (subsampled), shape is [B, H // 2, W // 2].
        output_cr (Tensor): Cr component (subsampled), shape is [B, H // 2, W // 2].
    """
    # Get components
    output_y: Tensor = input_ycbcr[:, 0]
    output_cb: Tensor = input_ycbcr[:, 1]
    output_cr: Tensor = input_ycbcr[:, 2]
    # Perform average pooling o Cb and Cr
    output_cb = F.avg_pool2d(output_cb[:, None], (2, 2))[:, 0]
    output_cr = F.avg_pool2d(output_cr[:, None], (2, 2))[:, 0]
    return output_y, output_cb, output_cr
