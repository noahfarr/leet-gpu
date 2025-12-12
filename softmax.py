import torch
import triton
import triton.language as tl

@triton.jit
def reduce_max_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input = tl.load(input_ptr + offsets, mask=mask, other=-float('inf'))
    max = tl.max(input, axis=0)
    tl.atomic_max(output_ptr, max)

@triton.jit
def reduce_sum_kernel(input_ptr, max_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    input = tl.load(input_ptr + offsets, mask=mask, other=-float('inf'))
    max = tl.load(max_ptr)
    sum = tl.sum(tl.exp(input - max), axis=0)
    tl.atomic_add(output_ptr, sum)

@triton.jit
def softmax_kernel(input_ptr, output_ptr, max_ptr, sum_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    max = tl.load(max_ptr)
    sum = tl.load(sum_ptr)

    input = tl.load(input_ptr + offsets, mask=mask, other=-float('inf'))
    softmax = tl.exp(input - max) / sum
    tl.store(output_ptr + offsets, softmax, mask=mask)

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    num_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    max = torch.full((), -float('inf'), device='cuda', dtype=torch.float32)
    reduce_max_kernel[(num_blocks,)](
        input, max, N,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    sum = torch.full((), 0, device='cuda', dtype=torch.float32)
    reduce_sum_kernel[(num_blocks,)](
        input, max, sum, N,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    softmax_kernel[(num_blocks,)](
        input, output,
        max, sum, N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
