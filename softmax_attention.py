import torch
import triton
import triton.language as tl

@triton.jit
def softmax_attention(Q_ptr, K_ptr, V_ptr, output_ptr,
                      M, N, d,
                      Q_stride_M, Q_stride_d,
                      K_stride_N, K_stride_d,
                      V_stride_N, V_stride_d,
                      output_stride_M, output_stride_d,
                      BLOCKSIZE_M: tl.constexpr,
                      BLOCKSIZE_N: tl.constexpr,
                      BLOCKSIZE_d: tl.constexpr):

    pid = tl.program_id(axis=0)

    offs_m = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_d = tl.arange(0, BLOCK_SIZE_d)

    Q_ptrs = Q_ptr + (offs_m[:, None] * Q_stride_m + offs_d[None, :] * Q_stride_d)
    Q_mask = offs_m[:, None] < M
    Q = tl.load(Q_ptrs, mask=Q_mask, other=0.0)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    running_sum = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
    running_max = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32) - float("inf")

    for n in range(0, N, BLOCK_SIZE_N):
        offs_n = n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        K_ptrs = K_ptr + (offs_d[None, :] * K_stride_d + offs_n[:, None] * K_stride_n)
        V_ptrs = V_ptr + (offs_d[None, :] * V_stride_d + offs_n[:, None] * V_stride_n)
            
        K = tl.load(K_ptrs, mask=offs_n[None, :] < N)
        V = tl.load(V_ptrs, mask=offs_n[None, :] < N)
        
        K_ptrs += BLOCK_SIZE_N * K_stride_n
        V_ptrs += BLOCK_SIZE_N * V_stride_n

        qk = tl.dot(Q, tl.trans(K)) / tl.sqrt(d)

        max = tl.max(qk, axis=1)
        logits = tl.exp(qk - m_j[:, None])
        sum = tl.sum(logits, 1)
        running_max = tl.maxiumum(running_max, max)


        qk = tl
        qkv = tl.dot(qk, V)


# Q, K, V, output are tensors on the GPU
def solve(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, output: torch.Tensor, M: int, N: int, d: int):    
    BLOCKSIZE_M = 16
    BLOCKSIZE_d = 128
    BLOCKSIZE_N = 64    
    grid = (triton.cdiv(M, BLOCKSIZE_M), triton.cdiv(d, BLOCKSIZE_d))
    
    Q_stride_M, Q_stride_d = Q.stride()
    K_stride_N, K_stride_d = K.stride()
    V_stride_N, V_stride_d = V.stride()
    out_stride_M, out_stride_d = output.stride()
    
    softmax_attention[grid](
        Q_ptr=Q,
        K_ptr=K, 
        V_ptr=V,
        output_ptr=output,
        
        M=M,
        N=N,
        d=d,
        
        Q_stride_M=Q_stride_M,
        Q_stride_d=Q_stride_d,
        K_stride_N=K_stride_N,
        K_stride_d=K_stride_d,
        V_stride_N=V_stride_N,
        V_stride_d=V_stride_d,

        output_stride_M=output_stride_M,
        output_stride_d=output_stride_d,
        
        BLOCKSIZE_M=BLOCKSIZE_M,
        BLOCKSIZE_d=BLOCKSIZE_d,
        BLOCKSIZE_N=BLOCKSIZE_N,
    )
