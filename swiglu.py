import torch
import triton
import triton.language as tl

@triton.jit
def swiglu(
    input, output, N, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mask = offsets < (N // 2)

    x = tl.load(input + offsets, mask=mask)
    y = tl.load(input + offsets + N//2, mask=mask)

    silu = x / (1 + tl.exp(-x))
    swiglu = silu * y

    tl.store(output + offsets, swiglu, mask=mask)

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N // 2, BLOCK_SIZE),)
    swiglu[grid](
        input, output, N, BLOCK_SIZE=BLOCK_SIZE
    )
