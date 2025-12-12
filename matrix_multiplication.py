import torch
import triton
import triton.language as tl

BLOCK_SIZE_M = tl.constexpr(32)
BLOCK_SIZE_N = tl.constexpr(32)
BLOCK_SIZE_K = tl.constexpr(32)

@triton.jit
def matrix_multiplication_kernel(
    a, b, c, 
    M, N, K,
    stride_am, stride_an, 
    stride_bn, stride_bk, 
    stride_cm, stride_ck
):
    pid_m = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_n = tl.arange(0, BLOCK_SIZE_N)

    a_ptrs = a + (offs_m[:, None] * stride_am + offs_n[None, :] * stride_an)
    b_ptrs = b + (offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    
    for k in range(0, N, BLOCK_SIZE_N):
        a_mask = (offs_m[:, None] < M) & ((k + offs_n[None, :]) < N)
        b_mask = ((k + offs_n[:, None]) < N) & (offs_k[None, :] < K)
        
        a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        accumulator = tl.dot(a_tile, b_tile, accumulator)
        
        a_ptrs += BLOCK_SIZE_N * stride_an
        b_ptrs += BLOCK_SIZE_N * stride_bn

    c_ptrs = c + (offs_m[:, None] * stride_cm + offs_k[None, :] * stride_ck)
    c_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    
    tl.store(c_ptrs, accumulator, mask=c_mask)
  
# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, M: int, N: int, K: int):
    stride_am, stride_an = N, 1 
    stride_bn, stride_bk = K, 1  
    stride_cm, stride_ck = K, 1
    
    grid = (M, K) 
    matrix_multiplication_kernel[grid](
        a, b, c,
        M, N, K,
        stride_am, stride_an,
        stride_bn, stride_bk,
        stride_cm, stride_ck
    )
