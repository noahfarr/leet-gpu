import torch
import triton
import triton.language as tl

@triton.jit
def reverse_kernel(
    input,
    N,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)

    offsets_a = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offsets_b = N - 1 - offsets_a

    mask = offsets_a < (N // 2)
    
    a_ptrs = tl.load(input + offsets_a, mask=mask)
    b_ptrs = tl.load(input + offsets_b, mask=mask)

    tl.store(input + offsets_a, b_ptrs, mask=mask)
    tl.store(input + offsets_b, a_ptrs, mask=mask)

# input is a tensor on the GPU
def solve(input: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    n_blocks = triton.cdiv(N // 2, BLOCK_SIZE)
    grid = (n_blocks,)
    
    reverse_kernel[grid](
        input,
        N,
        BLOCK_SIZE
    ) 
