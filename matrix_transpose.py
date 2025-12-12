import torch
import triton
import triton.language as tl

BLOCK_SIZE = tl.constexpr(1)

@triton.jit
def matrix_transpose_kernel(
    input, output,
    rows, cols,
    stride_ir, stride_ic,  
    stride_or, stride_oc
):
    pid_row, pid_col = tl.program_id(axis=0), tl.program_id(axis=1)

    row_offsets = pid_row * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    col_offsets = pid_col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    input_ptrs = input + (row_offsets[:, None] * stride_ir + col_offsets[:, None] * stride_ic)
    input_mask = (row_offsets[:, None] < rows) & (col_offsets[None, :] < cols)
    
    tile = tl.load(input_ptrs, mask=input_mask, other=0.0)

    transposed_tile = tl.trans(tile)

    output_ptrs = output + (col_offsets[:, None] * stride_or + row_offsets[None, :] * stride_oc)
    output_mask = (col_offsets[:, None] < cols) & (row_offsets[None, :] < rows)

    tl.store(output_ptrs, transposed_tile, mask=output_mask)

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, rows: int, cols: int):
    stride_ir, stride_ic = cols, 1  
    stride_or, stride_oc = rows, 1
    
    grid = (rows, cols)
    matrix_transpose_kernel[grid](
        input, output,
        rows, cols,
        stride_ir, stride_ic,
        stride_or, stride_oc
    ) 
