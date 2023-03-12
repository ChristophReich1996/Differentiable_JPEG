import torch
from torch import Tensor

from .utils import compression_strength_to_scale, QUANTIZATION_TABLE_C, QUANTIZATION_TABLE_Y
from .color import ycbcr_to_rgb, chroma_upsampling


def unpatchify_8x8(input: Tensor, H: int, W: int) -> Tensor:
    """Function reverses non-overlapping 8 x 8 patching.

    Args:
        input (Tensor): Input image of the shape [B, N, 8, 8].

    Returns:
        output (Tensor): Image patchify of the shape [B, H, W]
    """
    # Get input shape
    B, N, _, _ = input.shape  # type: int, int, int, int
    # Unpatch to [B, H, W]
    output: Tensor = input.view(B, H // 8, W // 8, 8, 8).permute(0, 1, 3, 2, 4).reshape(B, H, W)
    return output


def idct_8x8(input: Tensor) -> Tensor:
    """Performs a 8 x 8 discrete cosine transform.

    Args:
        input (Tensor): Patched input tensor of the shape [B, N, 8, 8].

    Returns:
        output (Tensor): DCT output tensor of the shape [B, N, 8, 8].
    """
    # Get dtype and device
    dtype: torch.dtype = input.dtype
    device: torch.device = input.device
    # Make and apply scaling
    alpha: Tensor = torch.ones(8, dtype=dtype, device=device)
    alpha[0] = 1.0 / (2**0.5)
    dct_scale: Tensor = torch.outer(alpha, alpha)
    input = input * dct_scale[None, None]
    # Make DCT tensor and scaling
    index: Tensor = torch.arange(8, dtype=dtype, device=device)
    x, y, u, v = torch.meshgrid(index, index, index, index)  # type: Tensor, Tensor, Tensor, TabError
    idct_tensor: Tensor = ((2 * u + 1) * x * torch.pi / 16).cos() * ((2 * v + 1) * y * torch.pi / 16).cos()
    # Apply DCT
    output: Tensor = 0.25 * torch.tensordot(input, idct_tensor, dims=2) + 128.0
    return output


def dequantize(
    input: Tensor,
    compression_strength: Tensor,
    quantization_table: Tensor,
) -> Tensor:
    """Function performs dequantization.

    Args:
        input (Tensor): Input tensor of the shape [B, N, 8, 8].
        compression_strength (Tensor): Compression strength to be applied, shape is [B].
        quantization_table (Tensor): Quantization table of the shape [8, 8].
        differentiable (bool): If true differentiable rounding is used. Default True.

    Returns:
        output (Tensor): Quantized output tensor of the shape [B, N, 8, 8].
    """
    # Perform scaling
    output: Tensor = input * (
        quantization_table[None, None] * compression_strength_to_scale(compression_strength)[:, None, None, None]
    )
    return output


def jpeg_decode(
    input_y: Tensor,
    input_cb: Tensor,
    input_cr: Tensor,
    compression_strength: Tensor,
    H: int,
    W: int,
) -> Tensor:
    """Performs JPEG decoding.

    Args:
        input_y (Tensor): Compressed Y component of the shape [B, N, 8, 8]
        input_cb (Tensor): Compressed Cb component of the shape [B, N, 8, 8]
        input_cr (Tensor): Compressed Cr component of the shape [B, N, 8, 8]
        compression_strength (Tensor): Compression strength of the shape [B].
        H (int): Original image height.
        W (int): Original image width.

    Returns:
        rgb_decoded (Tensor): Decompressed RGB image of the shape [B, 3, H, W].
    """
    assert isinstance(input_y, Tensor), f"Compressed Y component (input_y) must be a torch.Tensor, got {type(input_y)}."
    assert isinstance(
        input_cb, Tensor
    ), f"Compressed Cb component (input_cb) must be a torch.Tensor, got {type(input_cb)}."
    assert isinstance(
        input_cr, Tensor
    ), f"Compressed Cr component (input_cr) must be a torch.Tensor, got {type(input_cr)}."
    assert isinstance(
        compression_strength, Tensor
    ), f"Compression strength (compression_strength) must be a torch.Tensor, got {type(compression_strength)}."
    assert isinstance(H, int) and (H > 0), f"Height (H) must be as positive integer, got {H}."
    assert isinstance(W, int) and (W > 0), f"Width (W) must be as positive integer, got {H}."
    assert input_y.shape[0] == compression_strength.shape[0], (
        f"Batch size of Y components and compression strength must match, "
        f"got image shape {input_y.shape[0]} and compression strength shape {compression_strength.shape[0]}"
    )
    # Dequantize inputs
    input_y = dequantize(input_y, compression_strength, QUANTIZATION_TABLE_Y)
    input_cb = dequantize(input_cb, compression_strength, QUANTIZATION_TABLE_C)
    input_cr = dequantize(input_cr, compression_strength, QUANTIZATION_TABLE_C)
    # Perform inverse DCT
    idct_y: Tensor = idct_8x8(input_y)
    idct_cb: Tensor = idct_8x8(input_cb)
    idct_cr: Tensor = idct_8x8(input_cr)
    # Reverse patching
    image_y: Tensor = unpatchify_8x8(idct_y, H, W)
    image_cb: Tensor = unpatchify_8x8(idct_cb, H // 2, W // 2)
    image_cr: Tensor = unpatchify_8x8(idct_cr, H // 2, W // 2)
    # Perform chroma upsampling
    image_cb = chroma_upsampling(image_cb)
    image_cr = chroma_upsampling(image_cr)
    # Convert back into RGB space
    rgb_decoded: Tensor = ycbcr_to_rgb(torch.stack((image_y, image_cb, image_cr), dim=-1))
    rgb_decoded = rgb_decoded.permute(0, 3, 1, 2)
    return rgb_decoded
