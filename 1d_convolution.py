import torch
import triton
import triton.language as tl

@triton.jit
def conv1d_kernel(
    input, kernel, output,
    input_size, kernel_size,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)

    output_size = input_size - kernel_size + 1
    offsets_output = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_output = offsets_output < output_size

    accumulator = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for k in range(kernel_size):
        offsets_input = offsets_output + k
        mask_inputs = mask_output & offsets_input < input_size
        input_ptrs = tl.load(input + offsets_input, mask=mask_inputs, other=0.0)
        kernel_ptrs = tl.load(kernel + k)
        accumulator += input_ptrs * kernel_ptrs

    tl.store(output + offsets_output, accumulator, mask=mask_output)

# input, kernel, output are tensors on the GPU
def solve(input: torch.Tensor, kernel: torch.Tensor, output: torch.Tensor, input_size: int, kernel_size: int):
    BLOCK_SIZE = 1024
    n_blocks = triton.cdiv(input_size - kernel_size + 1, BLOCK_SIZE)
    grid = (n_blocks,)
    
    conv1d_kernel[grid](
        input, kernel, output,
        input_size, kernel_size,
        BLOCK_SIZE
    )
