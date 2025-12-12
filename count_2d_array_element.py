import torch
import triton
import triton.language as tl

@triton.jit
def count_equal_kernel(input_ptr, output_ptr, N, K, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    input = tl.load(input_ptr + offsets, mask=mask)

    counter = tl.sum(tl.where(input == K, 1, 0))

    tl.atomic_add(output_ptr, counter)

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int, M: int, K: int):
    BLOCK_SIZE = 256
    grid = lambda meta: (triton.cdiv(N * M, meta['BLOCK_SIZE']),)
    count_equal_kernel[grid](input, output, N * M, K, BLOCK_SIZE=BLOCK_SIZE)
