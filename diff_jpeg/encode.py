from typing import Tuple

import torch
from torch import Tensor

from .utils import compression_strength_to_scale, differentiable_rounding, QUANTIZATION_TABLE_C, QUANTIZATION_TABLE_Y
from .color import rgb_to_ycbcr, chroma_subsampling


def patchify_8x8(input: Tensor) -> Tensor:
    """Function extracts non-overlapping 8 x 8 patches from the given input image.

    Args:
        input (Tensor): Input image of the shape [B, H, W].

    Returns:
        output (Tensor): Image patchify of the shape [B, N, 8, 8]
    """
    # Get input shape
    B, H, W = input.shape  # type: int, int, int
    # Patchify to shape [B, N, H // 8, W // 8]
    output: Tensor = input.view(B, H // 8, 8, W // 8, 8).permute(0, 1, 3, 2, 4).reshape(B, -1, 8, 8)
    return output


def dct_8x8(input: Tensor) -> Tensor:
    """Performs a 8 x 8 discrete cosine transform.

    Args:
        input (Tensor): Patched input tensor of the shape [B, N, 8, 8].

    Returns:
        output (Tensor): DCT output tensor of the shape [B, N, 8, 8].
    """
    # Get dtype and device
    dtype: torch.dtype = input.dtype
    device: torch.device = input.device
    # Make DCT tensor and scaling
    index: Tensor = torch.arange(8, dtype=dtype, device=device)
    x, y, u, v = torch.meshgrid(index, index, index, index)  # type: Tensor, Tensor, Tensor, TabError
    dct_tensor: Tensor = ((2.0 * x + 1.0) * u * torch.pi / 16.0).cos() * ((2.0 * y + 1.0) * v * torch.pi / 16.0).cos()
    alpha: Tensor = torch.ones(8, dtype=dtype, device=device)
    alpha[0] = 1.0 / (2**0.5)
    dct_scale: Tensor = torch.einsum("i, j -> ij", alpha, alpha) * 0.25
    # Apply DCT
    output: Tensor = dct_scale[None, None] * torch.tensordot(input - 128.0, dct_tensor)
    return output


def quantize(
    input: Tensor,
    compression_strength: Tensor,
    quantization_table: Tensor,
    differentiable: bool = True,
) -> Tensor:
    """Function performs quantization.

    Args:
        input (Tensor): Input tensor of the shape [B, N, 8, 8].
        compression_strength (Tensor): Compression strength to be applied, shape is [B].
        quantization_table (Tensor): Quantization table of the shape [8, 8].
        differentiable (bool): If true differentiable rounding is used. Default True.

    Returns:
        output (Tensor): Quantized output tensor of the shape [B, N, 8, 8].
    """
    # Perform scaling
    output: Tensor = input / (
        quantization_table[None, None] * compression_strength_to_scale(compression_strength)[:, None, None, None]
    )
    # Perform rounding
    if differentiable:
        output = differentiable_rounding(output)
    else:
        output = torch.round(output)
    return output


def jpeg_encode(
    image_rgb: Tensor, compression_strength: Tensor, differentiable: bool = True
) -> Tuple[Tensor, Tensor, Tensor]:
    """Performs JPEG encoding.

    Args:
        image_rgb (Tensor): RGB input images of the shape [B, 3, H, W].
        compression_strength (Tensor): Compression strength of the shape [B].
        differentiable (bool): If true differentiable rounding is used. Default True.

    Returns:
        y_encoded (Tensor): Encoded Y component of the shape [B, N, 8, 8].
        cb_encoded (Tensor): Encoded Cb component of the shape [B, N, 8, 8].
        cr_encoded (Tensor): Encoded Cr component of the shape [B, N, 8, 8].
    """
    # Convert RGB image to YCbCr ans subsample
    image_ycbcr: Tensor = rgb_to_ycbcr(image_rgb.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    input_y, input_cb, input_cr = chroma_subsampling(image_ycbcr)  # type: Tensor, Tensor, Tensor
    # Patchify, DCT, and rounding
    input_y, input_cb, input_cr = patchify_8x8(input_y), patchify_8x8(input_cb), patchify_8x8(input_cr)
    dct_y, dct_cb, dct_cr = dct_8x8(input_y), dct_8x8(input_cb), dct_8x8(input_cr)  # type: Tensor, Tensor, Tensor
    y_encoded: Tensor = quantize(dct_y, compression_strength, QUANTIZATION_TABLE_Y, differentiable)
    cb_encoded: Tensor = quantize(dct_cb, compression_strength, QUANTIZATION_TABLE_C, differentiable)
    cr_encoded: Tensor = quantize(dct_cr, compression_strength, QUANTIZATION_TABLE_C, differentiable)
    return y_encoded, cb_encoded, cr_encoded
