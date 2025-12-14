import torch
import triton
import triton.language as tl

@triton.jit
def conv2d_kernel(
    input_ptr, kernel_ptr, output_ptr,
    input_row_stride, input_col_stride,
    kernel_row_stride, kernel_col_stride,
    output_row_stride, output_col_stride,
    input_rows, input_cols, 
    kernel_rows, kernel_cols, 
    output_rows, output_cols,
    BLOCK_SIZE_ROW: tl.constexpr,
    BLOCK_SIZE_COL: tl.constexpr,
):
    pid_row = tl.program_id(axis=0)
    pid_col= tl.program_id(axis=1)

    offs_output_row = pid_row * BLOCK_SIZE_ROW + tl.arange(0, BLOCK_SIZE_ROW)
    offs_output_col = pid_col * BLOCK_SIZE_COL + tl.arange(0, BLOCK_SIZE_COL)
    offs_output = (offs_output_row[:, None] * output_row_stride) + (offs_output_col[None, :] * output_col_stride)

    mask_output_row = offs_output_row < output_rows
    mask_output_col = offs_output_col < output_cols
    mask_output = mask_output_row[:, None] & mask_output_col[None, :]

    accumulator = tl.zeros([BLOCK_SIZE_ROW, BLOCK_SIZE_COL], dtype=tl.float32)

    for kernel_row in range(kernel_rows):
        for kernel_col in range(kernel_cols):
            offs_input_row = offs_output_row[:, None] + kernel_row
            offs_input_col = offs_output_col[None, :] + kernel_col
            offs_input = (offs_input_row * input_row_stride) + (offs_input_col * input_col_stride)

            mask_input_row = offs_input_row < input_rows
            mask_input_col = offs_input_col < input_cols
            mask_input = mask_input_row & mask_input_col
            
            input = tl.load(input_ptr + offs_input, mask=mask_input)

            offs_kernel = kernel_row * kernel_row_stride + kernel_col * kernel_col_stride
            kernel = tl.load(kernel_ptr + offs_kernel)

            accumulator += input * kernel

    tl.store(output_ptr + offs_output, accumulator, mask=mask_output)
        


# input, kernel, output are tensors on the GPU
def solve(input: torch.Tensor, kernel: torch.Tensor, output: torch.Tensor, input_rows: int, input_cols: int, kernel_rows: int, kernel_cols: int):
    BLOCK_SIZE = 32

    output_rows = input_rows - kernel_rows + 1
    output_cols = input_cols - kernel_cols + 1

    input = input.view(input_rows, input_cols)
    kernel = kernel.view(kernel_rows, kernel_cols)
    output = output.view(output_rows, output_cols)

    input_row_stride, input_col_stride = input.stride(0), input.stride(1),
    kernel_row_stride, kernel_col_stride = kernel.stride(0), kernel.stride(1),
    output_row_stride, output_col_stride = output.stride(0), output.stride(1),

    grid = (triton.cdiv(output_rows, BLOCK_SIZE), triton.cdiv(output_cols, BLOCK_SIZE))
    conv2d_kernel[grid](
        input, kernel, output,
        input_row_stride, input_col_stride,
        kernel_row_stride, kernel_col_stride,
        output_row_stride, output_col_stride,
        input_rows, input_cols,
        kernel_rows, kernel_cols,
        output_rows, output_cols,
        BLOCK_SIZE, BLOCK_SIZE
    )
