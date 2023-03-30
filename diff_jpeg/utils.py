import torch
from torch import Tensor

QUANTIZATION_TABLE_Y: Tensor = torch.tensor(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=torch.float,
)

QUANTIZATION_TABLE_C: Tensor = torch.tensor(
    [
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
    ],
    dtype=torch.float,
)


def compression_strength_to_scale(compression_strength: Tensor) -> Tensor:
    """Converts a given compression strength to the scaling factor.

    Args:
        compression_strength (Tensor): Compression strength ranging from 0 to 100. Any shape is supported.

    Returns:
        scale (Tensor): Scaling factor to be applied to quantization matrix. Same shape as input.
    """
    # Check the input is a tensor and in the correct range
    assert isinstance(
        compression_strength, Tensor
    ), f"Given compression strength must be a torch.Tensor, got {type(compression_strength)}."
    assert (compression_strength.max() <= 99.0) and (
        compression_strength.min() >= 1.0
    ), f"Compression strength must range from 0 to 100, got {compression_strength}."
    # Get scale
    scale: Tensor = torch.where(
        compression_strength < 50, 50.0 / compression_strength, 2.0 - (2.0 * compression_strength / 100.0)
    )
    return scale


def differentiable_rounding(input: Tensor) -> Tensor:
    """This function implements differentiable rounding.

    Args:
        input (Tensor): Input tensor of any shape to be rounded.

    Returns:
        output (Tensor): Pseudo rounded tensor of the same shape as input tensor.
    """
    # Perform differentiable rounding
    output: Tensor = torch.round(input) + (input - torch.round(input)) ** 3
    return output
