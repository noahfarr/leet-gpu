import torch
import triton
import triton.language as tl


@triton.jit
def partial_sum(
    data_ptr, output_ptr, sums_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    data = tl.load(data_ptr + offsets, mask=mask, other=0.0)

    sum = tl.sum(data)
    tl.store(sums_ptr + pid, sum)

    cumsum = tl.cumsum(data)
    tl.store(output_ptr + offsets, cumsum, mask=mask)

@triton.jit
def prefix_sum(
    output_ptr, sums_ptr,
    N, 
    BLOCK_SIZE: tl.constexpr, NUM_BLOCKS: tl.constexpr
):
    pid = tl.program_id(axis=0)

    if pid == 0:
        return

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    data = tl.load(output_ptr + offsets, mask=mask)
    partial_sum = tl.load(sums_ptr + pid - 1)

    prefix_sum = data + partial_sum 
    tl.store(output_ptr + offsets, prefix_sum, mask)

# data and output are tensors on the GPU
def solve(data: torch.Tensor, output: torch.Tensor, n: int):
    BLOCK_SIZE = 128
    NUM_BLOCKS = triton.cdiv(n, BLOCK_SIZE)
    grid = (NUM_BLOCKS,)
    sums= torch.zeros(grid, dtype=torch.float32, device='cuda')
    partial_sum[grid](
        data, output, sums,
        n,
        BLOCK_SIZE,
    )
    
    prefix_sum[grid](
        output, torch.cumsum(sums, dim=0),
        n,
        BLOCK_SIZE, NUM_BLOCKS
    )
