import torch
import triton
import triton.language as tl


@triton.jit
def reduce_kernel(input_ptr, output_ptr, BLOCK_SIZE: tl.constexpr,  N: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    input = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    sum = tl.sum(input)

    tl.atomic_add(output_ptr, sum, sem="relaxed")

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE), )
    reduce_kernel[grid](input, output, BLOCK_SIZE, N)
