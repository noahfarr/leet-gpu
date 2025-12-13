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
                      BLOCK_SIZE_M: tl.constexpr,
                      BLOCK_SIZE_N: tl.constexpr,
                      BLOCK_SIZE_d: tl.constexpr):

    pid = tl.program_id(axis=0)

    offs_m = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_d = tl.arange(0, BLOCK_SIZE_d)

    Q_ptrs = Q_ptr + (offs_m[:, None] * Q_stride_M + offs_d[None, :] * Q_stride_d)
    Q_mask = (offs_m[:, None] < M) & (offs_d[None, :] < d)
    Q = tl.load(Q_ptrs, mask=Q_mask, other=0.0)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_d), dtype=tl.float32)
    running_sum = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
    running_max = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32) - float("inf")

    for n in range(0, N, BLOCK_SIZE_N):
        offs_n = n + tl.arange(0, BLOCK_SIZE_N)
        
        K_ptrs = K_ptr + (offs_d[None, :] * K_stride_d + offs_n[:, None] * K_stride_N)
        V_ptrs = V_ptr + (offs_d[None, :] * V_stride_d + offs_n[:, None] * V_stride_N)
            
        mask_K = (offs_n[:, None] < N) & (offs_d[None, :] < d)
        mask_V = (offs_n[:, None] < N) & (offs_d[None, :] < d)
        
        K = tl.load(K_ptrs, mask=mask_K, other=0.0)
        V = tl.load(V_ptrs, mask=mask_V, other=0.0)

        qk = tl.dot(Q, tl.trans(K)) / tl.sqrt(d.to(tl.float32))
        qk = tl.where(offs_n[None, :] < N, qk, float("-inf"))

        max = tl.max(qk, axis=1)
        logits = tl.exp(qk - max[:, None])
        sum = tl.sum(logits, 1)

        difference = running_max - tl.maximum(running_max, max)
        running_max = tl.maximum(running_max, max)
        alpha = tl.exp(difference)
        beta = tl.exp(max - running_max)

        running_sum = running_sum * alpha + sum * beta

        qkv = tl.dot(logits, V)

        accumulator = accumulator * alpha[:, None] + qkv * beta[:, None]

    output = accumulator / running_sum[:, None]
    
    output_ptrs = output_ptr + (offs_m[:, None] * output_stride_M + offs_d[None, :] * output_stride_d)
    output_mask = (offs_m[:, None] < M) & (offs_d[None, :] < d)
    tl.store(output_ptrs, output, mask=output_mask)


def solve(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, output: torch.Tensor, M: int, N: int, d: int):    
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_d = max(16, triton.next_power_of_2(d))
    BLOCK_SIZE_N = 32

    grid = (triton.cdiv(M, BLOCK_SIZE_M), )
    
    Q_stride_M, Q_stride_d = Q.stride()
    K_stride_N, K_stride_d = K.stride()
    V_stride_N, V_stride_d = V.stride()
    output_stride_M, output_stride_d = output.stride()
    
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
        
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_d=BLOCK_SIZE_d,
        BLOCK_SIZE_N=BLOCK_SIZE_N,

        num_stages=1,
        num_warps=4
    )
