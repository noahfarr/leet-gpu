import torch
import triton
import triton.language as tl

@triton.jit
def invert_kernel(
    image,
    width, height,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = (offsets < width * height * 4) & (offsets % 4 != 3)

    image_ptrs = tl.load(image + offsets, mask=mask)
    tl.store(image + offsets, 255 - image_ptrs, mask=mask)

# image is a tensor on the GPU
def solve(image: torch.Tensor, width: int, height: int):
    BLOCK_SIZE = 1024
    n_pixels = width * height * 4
    grid = (triton.cdiv(n_pixels, BLOCK_SIZE),)
    
    invert_kernel[grid](
        image,
        width, height,
        BLOCK_SIZE
    ) 
