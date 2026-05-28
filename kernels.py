"""STUDENT FILE: implement the Triton kernels and pipeline drivers.

You implement:
  - Six @triton.jit kernels: f1_kernel, f2_kernel, transpose_kernel,
    f4_kernel_L2, dft_kernel, bailey_scale_kernel.
  - The f1_launch and f2_launch grid-choice wrappers around them.
  - The pipeline drivers: f3_launch, f5_launch, _f6_rec, _f7_rec.
  - f6_factor: the chunk-recipe for F6/F7.

You do NOT implement (left given below):
  - The thin launch wrappers _transpose, _fft_chunk, _scale, _lookup_tw.
    These are mechanical "pick the grid and launch one kernel" helpers.
  - The tuning constants F4_L2_BLOCK_B, DFT_BLOCK_B, SCALE_BLOCK,
    TRANSPOSE_BLOCK.

The signatures below are the ones the harness calls -- your job is to fill
the bodies. When your code passes sanity_check.py, you're done.
"""

import math

import torch
import triton
import triton.language as tl


# Tunings -- GIVEN.
F4_L2_BLOCK_B = 2
DFT_BLOCK_B = 16
SCALE_BLOCK = 32
TRANSPOSE_BLOCK = 32


# =============================================================================
# Device-function helper: complex matmul
# =============================================================================
# Implement this once -- f1_kernel, f4_kernel_L2, and dft_kernel all call it.


@triton.jit
def _cdot(a_re, a_im, b_re, b_im):
    """Complex matmul Y = A @ B as four real tl.dot calls.

    Returns (y_re, y_im) in fp32 (out_dtype=tl.float32). Caller is responsible
    for any fp16 down-cast on store. Works at any matmul shape tl.dot accepts.

    Used by f1_kernel, f4_kernel_L2, and dft_kernel. Don't reimplement the
    four-tl.dot expansion at each call site -- implement once here, call
    everywhere.

    TODO: implement.
    """
    y_re = tl.dot(a_re, b_re, out_dtype=tl.float32) - tl.dot(a_im, b_im, out_dtype=tl.float32)
    y_im = tl.dot(a_re, b_im, out_dtype=tl.float32) + tl.dot(a_im, b_re, out_dtype=tl.float32)
    return y_re, y_im


# =============================================================================
# Chunk factorization for F6 / F7
# =============================================================================

def f6_factor(N: int) -> list[int]:
    """Factor N = 2^k into FFT chunks.

    Recipe: prefer 256-length chunks (radix-256, handled by f4_kernel_L2), then
    16-length (handled by dft_kernel via the padded radix-16 path), then a
    small leftover in {2, 4, 8} for the remaining bits. chunks[0] is the
    innermost (fastest) input axis. Examples:
        256 -> [256]                4096 -> [256, 16]
        65536 -> [256, 256]         1048576 -> [256, 256, 16]
        64 -> [16, 4]               2 -> [2]
    """
    # TODO
    assert N >= 2 and (N & (N - 1)) == 0, f"N must be a power of 2 >= 2; got {N}"
    k = N.bit_length() - 1
    n256, rb = divmod(k, 8)
    n16, rb2 = divmod(rb, 4)
    rsmall = 1 << rb2
    chunks = [256] * n256 + [16] * n16 + ([rsmall] if rsmall > 1 else [])
    assert math.prod(chunks) == N
    return chunks


f7_factor = f6_factor   # F7 reuses F6's chunk recipe


# =============================================================================
# F1: DFT as one dense complex matmul (four tl.dot)
# =============================================================================

@triton.jit
def f1_kernel(
    x_re_ptr, x_im_ptr,    # (B, N) fp16
    W_re_ptr, W_im_ptr,    # (N, N) fp16; W[n, k]
    y_re_ptr, y_im_ptr,    # (B, N) fp32
    B,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Y = X @ W^T as four (BLOCK_M, BLOCK_K) x (BLOCK_K, BLOCK_N) tl.dot calls.

    Y[b, n] = sum_k X[b, k] * W[n, k]. Load W in transposed access
    (W_T[k, n] = W[n, k]) so tl.dot reads it the way it wants.

    Use `_cdot(x_re, x_im, W_T_re, W_T_im)` for the per-block complex matmul;
    accumulate its fp32 output into `acc_re` / `acc_im`.

    Dtype contract (same as F4): loads are fp16, `tl.dot` runs with
    `out_dtype=tl.float32` (handled by `_cdot`), accumulator is fp32, store
    is fp32. Allocations in `f1_alloc` already match this -- x_re/x_im are
    fp16, y_re/y_im are fp32.

    TODO: implement.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
 
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
 
    acc_re = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_im = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
 
    for k_start in range(0, N, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
 
        # Load x block: (BLOCK_M, BLOCK_K)
        x_mask = (offs_m[:, None] < B) & (offs_k[None, :] < N)
        x_re = tl.load(x_re_ptr + offs_m[:, None] * N + offs_k[None, :], mask=x_mask, other=0.0).to(tl.float16)
        x_im = tl.load(x_im_ptr + offs_m[:, None] * N + offs_k[None, :], mask=x_mask, other=0.0).to(tl.float16)
 
        # Load W^T block: W[offs_n, offs_k] -> shape (BLOCK_N, BLOCK_K), then transpose to (BLOCK_K, BLOCK_N)
        w_mask = (offs_n[:, None] < N) & (offs_k[None, :] < N)
        wt_re = tl.load(W_re_ptr + offs_n[:, None] * N + offs_k[None, :], mask=w_mask, other=0.0).to(tl.float16)
        wt_im = tl.load(W_im_ptr + offs_n[:, None] * N + offs_k[None, :], mask=w_mask, other=0.0).to(tl.float16)
        # Transpose: (BLOCK_N, BLOCK_K) -> (BLOCK_K, BLOCK_N)
        wt_re = tl.trans(wt_re)
        wt_im = tl.trans(wt_im)
 
        blk_re, blk_im = _cdot(x_re, x_im, wt_re, wt_im)
        acc_re += blk_re
        acc_im += blk_im
 
    out_mask = (offs_m[:, None] < B) & (offs_n[None, :] < N)
    tl.store(y_re_ptr + offs_m[:, None] * N + offs_n[None, :], acc_re, mask=out_mask)
    tl.store(y_im_ptr + offs_m[:, None] * N + offs_n[None, :], acc_im, mask=out_mask)


def f1_launch(x_re, x_im, W_re, W_im, y_re, y_im):
    """Grid: (cdiv(B, BLOCK_M), cdiv(N, BLOCK_N)). One program tiles a
    (BLOCK_M, BLOCK_N) output square. tl.dot needs all three dims >=16, so B
    should be >= 16.

    TODO: implement.
    """
    B, N = x_re.shape
    BLOCK_M = 16
    BLOCK_K = N  # full K in one shot (N <= 256)
    BLOCK_N = 16
    grid = (triton.cdiv(B, BLOCK_M), triton.cdiv(N, BLOCK_N))
    f1_kernel[grid](
        x_re, x_im, W_re, W_im, y_re, y_im,
        B, N=N, BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N,
    )


# =============================================================================
# F2: radix-2 Cooley-Tukey, single program per signal
# =============================================================================
# F3 reuses this kernel! For F2, only BAILEY_EPILOGUE=False, STRIDED_STORE=False need to be implemented.
#
# Call-site cheatsheet:
#   F2 vanilla:  pid -> one signal in (B, N). Grid: (B,).
#                BAILEY_EPILOGUE=False, STRIDED_STORE=False.
#                OUTER_DIM and N_TOTAL unused (pass 1 / 0).
#                bt_*_ptr: pass tw_*_ptr again (sentinel; never read).
#   F2-A (F3):   pid -> (b, n1). Grid: (B*N1,). FFT length N=N2.
#                BAILEY_EPILOGUE=True, STRIDED_STORE=False.
#                OUTER_DIM=N1 (n1 = pid % N1).
#                bt_*_ptr: real Bailey twiddles shape (N1, N2).
#   F2-B (F3):   pid -> (b, k2). Grid: (B*N2,). FFT length N=N1.
#                BAILEY_EPILOGUE=False, STRIDED_STORE=True.
#                OUTER_DIM=N2, N_TOTAL=N1*N2.
#                bt_*_ptr: sentinel.

@triton.jit
def f2_kernel(
    x_re_ptr, x_im_ptr,        # (B, N) fp32 input
    y_re_ptr, y_im_ptr,        # (B, N) fp32 output (layout depends on STRIDED_STORE)
    tw_re_ptr, tw_im_ptr,      # (N/2,) fp32 radix-2 twiddles
    perm_ptr,                   # (N,) int32 bit-reversal index
    bt_re_ptr, bt_im_ptr,       # (OUTER_DIM, N) fp32 Bailey twiddles (BAILEY_EPILOGUE only)
    OUTER_DIM, N_TOTAL,
    N: tl.constexpr,
    LOG2_N: tl.constexpr,
    BAILEY_EPILOGUE: tl.constexpr,
    STRIDED_STORE: tl.constexpr,
):
    """Radix-2 Cooley-Tukey FFT in registers, with optional Bailey epilogue and
    strided store. log2(N) butterfly stages via tl.gather for partner shuffle.

    TODO: implement.
    """
    pid = tl.program_id(0)
 
    # Determine which signal / outer index this pid handles
    if BAILEY_EPILOGUE:
        # pid = b * OUTER_DIM + n1  (grid: B*OUTER_DIM)
        n1 = pid % OUTER_DIM
        b = pid // OUTER_DIM
        # Input is laid out as (B, OUTER_DIM, N) in F3 after T1
        base_in = b * OUTER_DIM * N + n1 * N
    elif STRIDED_STORE:
        # pid = b * OUTER_DIM + k2  (grid: B*OUTER_DIM) for F2-B
        k2 = pid % OUTER_DIM
        b = pid // OUTER_DIM
        base_in = b * OUTER_DIM * N + k2 * N
    else:
        b = pid
        base_in = b * N
 
    idx = tl.arange(0, N)
 
    # Load with bit-reversal permutation
    perm = tl.load(perm_ptr + idx)
    v_re = tl.load(x_re_ptr + base_in + perm)
    v_im = tl.load(x_im_ptr + base_in + perm)
 
    # Butterfly stages
    for s in tl.static_range(LOG2_N):
        half = 1 << s
        # tw_idx for each element
        tw_idx = (idx & (half - 1)) * (N >> (s + 1))
        tw_re = tl.load(tw_re_ptr + tw_idx)
        tw_im = tl.load(tw_im_ptr + tw_idx)
 
        # Partner index: j XOR 2^s
        partner = idx ^ half
 
        p_re = tl.gather(v_re, partner, 0)
        p_im = tl.gather(v_im, partner, 0)
 
        # Complex multiply: w * p
        wp_re = tw_re * p_re - tw_im * p_im
        wp_im = tw_re * p_im + tw_im * p_re
 
        # Butterfly: if (idx & half) == 0: v = v + wp, else v = v - wp (from partner's perspective)
        is_lo = (idx & half) == 0
        new_re = tl.where(is_lo, v_re + wp_re, v_re - wp_re)
        new_im = tl.where(is_lo, v_im + wp_im, v_im - wp_im)
        v_re = new_re
        v_im = new_im
 
    # Bailey epilogue: multiply by cross-twiddle bt[n1, k2]
    if BAILEY_EPILOGUE:
        bt_re = tl.load(bt_re_ptr + n1 * N + idx)
        bt_im = tl.load(bt_im_ptr + n1 * N + idx)
        out_re = v_re * bt_re - v_im * bt_im
        out_im = v_re * bt_im + v_im * bt_re
        v_re = out_re
        v_im = out_im
 
    # Store
    if STRIDED_STORE:
        # Output layout: (B, N1, N2) with N1=OUTER_DIM, N2=N
        # pid = b * OUTER_DIM + k2, signal k2 writes to column k2 of the (N1, N2) matrix
        # Output index for output element k1: b*N_TOTAL + k1*OUTER_DIM + k2
        k2 = pid % OUTER_DIM
        b = pid // OUTER_DIM
        out_idx = b * N_TOTAL + idx * OUTER_DIM + k2
        tl.store(y_re_ptr + out_idx, v_re)
        tl.store(y_im_ptr + out_idx, v_im)
    else:
        if BAILEY_EPILOGUE:
            tl.store(y_re_ptr + pid * N + idx, v_re)
            tl.store(y_im_ptr + pid * N + idx, v_im)
        else:
            tl.store(y_re_ptr + b * N + idx, v_re)
            tl.store(y_im_ptr + b * N + idx, v_im)



def f2_launch(x_re, x_im, y_re, y_im, tw_re, tw_im, perm):
    """Grid: (B,). One program per length-N signal. Vanilla mode.

    TODO: implement.
    """
    B, N = x_re.shape
    LOG2_N = int(math.log2(N))
    f2_kernel[(B,)](
        x_re, x_im, y_re, y_im, tw_re, tw_im, perm,
        tw_re, tw_im,  # sentinel bt ptrs (never read)
        1, 0,
        N=N, LOG2_N=LOG2_N,
        BAILEY_EPILOGUE=False, STRIDED_STORE=False,
    )


# =============================================================================
# transpose_kernel: (B, R, C) -> (B, C, R), paired re/im
# =============================================================================

@triton.jit
def transpose_kernel(
    x_re_ptr, x_im_ptr,     # (B*R*C,) fp16 or fp32 input
    y_re_ptr, y_im_ptr,     # (B*R*C,) fp16 or fp32 output
    R, C,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Logical (B, R, C) -> (B, C, R) transpose. Grid: (cdiv(R, BLOCK_R),
    cdiv(C, BLOCK_C), B). Each program copies a (BLOCK_R, BLOCK_C) tile.

    TODO: implement.
    """
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    b = tl.program_id(2)
 
    offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
 
    base = b * R * C
    # Input: (B, R, C) -> index: base + r*C + c
    mask = (offs_r[:, None] < R) & (offs_c[None, :] < C)
    in_idx = base + offs_r[:, None] * C + offs_c[None, :]
    tile_re = tl.load(x_re_ptr + in_idx, mask=mask, other=0.0)
    tile_im = tl.load(x_im_ptr + in_idx, mask=mask, other=0.0)
 
    # Output: (B, C, R) -> index: base + c*R + r
    out_idx = base + offs_c[None, :] * R + offs_r[:, None]
    tl.store(y_re_ptr + out_idx, tile_re, mask=mask)
    tl.store(y_im_ptr + out_idx, tile_im, mask=mask)


# =============================================================================
# F4: tcFFT radix-16 single-program FFT (N = 256, L = 2)
# =============================================================================
# See the kernel docstring for the tl.permute tuple-literal gotcha.

@triton.jit
def f4_kernel_L2(
    x_re_ptr, x_im_ptr,    # (B, 256) fp16
    y_re_ptr, y_im_ptr,    # (B, 256) or (B//M, 256, M) fp16
    F_re_ptr, F_im_ptr,    # (16, 16) fp16 -- F_16 DFT matrix
    tw_re_ptr, tw_im_ptr,  # (L=2, 16, 16) fp16 stacked stage twiddles
    B, M,
    BLOCK_B: tl.constexpr,
    STAGE_STOP: tl.constexpr,
    STORE_T: tl.constexpr,
):
    """tcFFT length-256 FFT as two stages of (permute + per-stage twiddle +
    length-16 DFT via four tl.dot). fp16 storage, fp32 matmul accumulators.

    `STAGE_STOP` and `M` are both degenerate in vanilla F4 (`STAGE_STOP=L=2`,
    `M=1`). They exist so the same kernel handles two extra uses:
      - `STAGE_STOP=1`: stop after the s=0 stage, for the sanity_check.py
        stage-1 isolation test (no twiddles, no second matmul).
      - `M>1` with `STORE_T=True`: F7's fused FFT-m_0+T3, writing the
        transposed (rows_outer, 256, M) layout the next level expects.

    STORE_T=False (M=1): natural (B, 256) row-major output.
    STORE_T=True  (M>1): transposed (B//M, 256, M) output for F7 fusion.

    Each stage's four-`tl.dot` is one `_cdot` call; cast its fp32 output to
    fp16 before the next stage.

    Dtype contract:
        Loads:           fp16
        Reshape/permute: fp16 (free)
        tl.dot inputs:   fp16, out_dtype=tl.float32  (use _cdot)
        Twiddle mul:     fp32 * fp16 -> fp32
        Inter-stage:     .to(tl.float16) before next iter's reshape
        Store:           fp16
    Forgetting the inter-stage cast doubles register pressure and passes the
    L=2 tolerance, but fails as soon as F6 stacks more stages.

    Triton 3.6 gotcha -- tl.permute requires LITERAL tuples:
        tl.permute(x, (1, 0, 2))                  # works
        perm = (1, 0, 2); tl.permute(x, perm)     # fails
    Inline each stage's permute tuple at the call site; don't store the
    schedule in a loop variable.

    TODO: implement.
    """
    pid = tl.program_id(0)
    b_start = pid * BLOCK_B
    b_offs = b_start + tl.arange(0, BLOCK_B)
    n_offs = tl.arange(0, 256)
 
    # Load F (16x16 DFT matrix)
    F_re = tl.load(F_re_ptr + tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :])
    F_im = tl.load(F_im_ptr + tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :])
 
    # Load input: (BLOCK_B, 256) fp16
    mask = b_offs[:, None] < B
    x_re = tl.load(x_re_ptr + b_offs[:, None] * 256 + n_offs[None, :], mask=mask, other=0.0)
    x_im = tl.load(x_im_ptr + b_offs[:, None] * 256 + n_offs[None, :], mask=mask, other=0.0)
 
    # Reshape to (BLOCK_B, 16, 16): x[b, d0, d1] with d0=high digit, d1=low digit
    # In memory: n = d0*16 + d1, so tile[b, d0, d1] = x[b, d0*16 + d1]
    x_re = tl.reshape(x_re, (BLOCK_B, 16, 16))
    x_im = tl.reshape(x_im, (BLOCK_B, 16, 16))
 
    # Stage s=0: permute so axis 0 (d0) is first (already there), DFT along d0
    # Permute: (BLOCK_B, d0, d1) -> (BLOCK_B, d0, d1) [no change at s=0, d0 is axis 1]
    # Actually we need to bring axis s to position 1 (first of the 2 tile dims)
    # At s=0: tile is (BLOCK_B, d0, d1), we want d0 at position 1 -> already there
    # DFT along axis 1 (d0): matmul (16, BLOCK_B*16) vs (16, 16) F
    # Reshape to (16, BLOCK_B*16) for the matmul, apply F along rows
    tile_re = tl.reshape(x_re, (16, BLOCK_B * 16))
    tile_im = tl.reshape(x_im, (16, BLOCK_B * 16))
    # y = F @ x: (16, 16) @ (16, BLOCK_B*16) -> (16, BLOCK_B*16)
    out_re, out_im = _cdot(F_re, F_im, tile_re, tile_im)
    # Cast to fp16 for next stage
    out_re = out_re.to(tl.float16)
    out_im = out_im.to(tl.float16)
    # Reshape back to (BLOCK_B, 16, 16): now axis 1 = e1 (output digit for s=0: e_{L-1-0}=e_1)
    # After s=0 stage, tile is (e1, BLOCK_B, d1) -> need to reshape to (BLOCK_B, e1, d1)
    out_re = tl.reshape(out_re, (16, BLOCK_B, 16))
    out_im = tl.reshape(out_im, (16, BLOCK_B, 16))
    # Permute to (BLOCK_B, e1, d1)
    out_re = tl.permute(out_re, (1, 0, 2))
    out_im = tl.permute(out_im, (1, 0, 2))
 
    if STAGE_STOP == 1:
        # Store result after stage 0 only: flatten (BLOCK_B, 16, 16) -> (BLOCK_B, 256)
        out_re = tl.reshape(out_re, (BLOCK_B, 256))
        out_im = tl.reshape(out_im, (BLOCK_B, 256))
        tl.store(y_re_ptr + b_offs[:, None] * 256 + n_offs[None, :], out_re, mask=mask)
        tl.store(y_im_ptr + b_offs[:, None] * 256 + n_offs[None, :], out_im, mask=mask)
    else:
        # Stage s=1: bring d1 (axis 2) to position 1
        # Tile is (BLOCK_B, e1, d1), permute to (BLOCK_B, d1, e1)
        out_re = tl.permute(out_re, (0, 2, 1))
        out_im = tl.permute(out_im, (0, 2, 1))
 
        # Load stage-1 twiddles: tw[1, m, c] shape (16, 16)
        # m = d1 index (position 1 in tile), c = e1 index (position 2)
        tw1_re = tl.load(tw_re_ptr + 1 * 16 * 16 + tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :])
        tw1_im = tl.load(tw_im_ptr + 1 * 16 * 16 + tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :])
 
        # Apply twiddle: tile shape (BLOCK_B, d1=16, e1=16)
        # tw1[d1, e1] = tw[1, m=d1, c=e1]
        # Multiply: (BLOCK_B, 16, 16) elementwise with (16, 16) broadcast over batch
        t_re = tl.reshape(out_re, (BLOCK_B, 16, 16))
        t_im = tl.reshape(out_im, (BLOCK_B, 16, 16))
        # twiddle broadcast: (1, 16, 16)
        tw1_re_b = tl.reshape(tw1_re, (1, 16, 16))
        tw1_im_b = tl.reshape(tw1_im, (1, 16, 16))
        # complex multiply
        tmp_re = t_re.to(tl.float32) * tw1_re_b.to(tl.float32) - t_im.to(tl.float32) * tw1_im_b.to(tl.float32)
        tmp_im = t_re.to(tl.float32) * tw1_im_b.to(tl.float32) + t_im.to(tl.float32) * tw1_re_b.to(tl.float32)
        t_re = tmp_re.to(tl.float16)
        t_im = tmp_im.to(tl.float16)
 
        # DFT along axis 1 (d1): (BLOCK_B, d1, e1) -> F @ each (d1, e1) slice
        # Reshape to (16, BLOCK_B*16) for matmul
        t_re_flat = tl.reshape(t_re, (16, BLOCK_B * 16))
        t_im_flat = tl.reshape(t_im, (16, BLOCK_B * 16))
        out2_re, out2_im = _cdot(F_re, F_im, t_re_flat, t_im_flat)
        out2_re = out2_re.to(tl.float16)
        out2_im = out2_im.to(tl.float16)
        # Shape is now (e0=16, BLOCK_B*e1=BLOCK_B*16)
        # Reshape to (e0, BLOCK_B, e1) then permute to (BLOCK_B, e0, e1)
        out2_re = tl.reshape(out2_re, (16, BLOCK_B, 16))
        out2_im = tl.reshape(out2_im, (16, BLOCK_B, 16))
        out2_re = tl.permute(out2_re, (1, 0, 2))
        out2_im = tl.permute(out2_im, (1, 0, 2))
        # Now output is (BLOCK_B, e0, e1) where k = e0*16 + e1
 
        if STORE_T:
            # Fused FFT-m0+T3 output layout: (B//M, 256, M)
            # Each row b corresponds to outer_b = b // M, m_idx = b % M
            # Output: y[outer_b, k, m_idx]
            out_flat = tl.reshape(out2_re, (BLOCK_B, 256))
            out_flat_im = tl.reshape(out2_im, (BLOCK_B, 256))
            # b_offs: actual batch indices
            outer_b = b_offs // M
            m_idx = b_offs % M
            # B_outer = B // M, output shape: (B//M, 256, M)
            B_outer = B // M
            out_idx = outer_b[:, None] * (256 * M) + n_offs[None, :] * M + m_idx[:, None]
            tl.store(y_re_ptr + out_idx, out_flat, mask=mask)
            tl.store(y_im_ptr + out_idx, out_flat_im, mask=mask)
        else:
            out2_re = tl.reshape(out2_re, (BLOCK_B, 256))
            out2_im = tl.reshape(out2_im, (BLOCK_B, 256))
            tl.store(y_re_ptr + b_offs[:, None] * 256 + n_offs[None, :], out2_re, mask=mask)
            tl.store(y_im_ptr + b_offs[:, None] * 256 + n_offs[None, :], out2_im, mask=mask)


# =============================================================================
# dft_kernel: padded length-R DFT for the small chunks (R in {2, 4, 8, 16})
# =============================================================================

@triton.jit
def dft_kernel(
    x_re_ptr, x_im_ptr,     # (rows, R) fp16
    y_re_ptr, y_im_ptr,     # (rows, R) or (rows//M, R, M) fp16
    M_re_ptr, M_im_ptr,     # (16, 16) fp16 padded-R DFT matrix
    rows, M,
    R: tl.constexpr,
    BLOCK_B: tl.constexpr,
    STORE_T: tl.constexpr,
):
    """Padded length-R DFT via a (16, 16) tl.dot. STORE_T toggles natural
    vs transposed output (same pattern as f4_kernel_L2).

    One `_cdot(x_re, x_im, MT_re, MT_im)` call replaces the four `tl.dot`
    expansions; cast its fp32 result to fp16 on store.

    TODO: implement.
    """
    pid = tl.program_id(0)
    b_start = pid * BLOCK_B
    b_offs = b_start + tl.arange(0, BLOCK_B)
    mask_b = b_offs < rows
 
    r_offs = tl.arange(0, 16)
 
    # Load padded input: (BLOCK_B, 16) with R valid columns
    x_mask = mask_b[:, None] & (r_offs[None, :] < R)
    x_re = tl.load(x_re_ptr + b_offs[:, None] * R + r_offs[None, :], mask=x_mask, other=0.0)
    x_im = tl.load(x_im_ptr + b_offs[:, None] * R + r_offs[None, :], mask=x_mask, other=0.0)
 
    # Load (16, 16) DFT matrix
    MT_re = tl.load(M_re_ptr + tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :])
    MT_im = tl.load(M_im_ptr + tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :])
 
    # y = x @ MT^T: (BLOCK_B, 16) @ (16, 16) -> (BLOCK_B, 16)
    # Actually y = x @ M^T where M[j,k] = F[j,k]; since F is symmetric in index structure
    # We want y[b, k] = sum_n x[b, n] * F[k, n] = (x @ F^T)[b, k]
    # MT_re is already F, so we need F^T
    MT_re_T = tl.trans(MT_re)
    MT_im_T = tl.trans(MT_im)
    y_re, y_im = _cdot(x_re, x_im, MT_re_T, MT_im_T)
 
    # Take first R outputs
    y_re = y_re.to(tl.float16)
    y_im = y_im.to(tl.float16)
 
    r_valid = tl.arange(0, 16)
    out_mask = mask_b[:, None] & (r_valid[None, :] < R)
 
    if STORE_T:
        outer_b = b_offs // M
        m_idx = b_offs % M
        out_idx = outer_b[:, None] * (R * M) + r_valid[None, :] * M + m_idx[:, None]
        tl.store(y_re_ptr + out_idx, y_re, mask=out_mask)
        tl.store(y_im_ptr + out_idx, y_im, mask=out_mask)
    else:
        tl.store(y_re_ptr + b_offs[:, None] * R + r_valid[None, :], y_re, mask=out_mask)
        tl.store(y_im_ptr + b_offs[:, None] * R + r_valid[None, :], y_im, mask=out_mask)


# =============================================================================
# bailey_scale_kernel: elementwise w_N^{n1 kM} multiply with optional fused T2
# =============================================================================

@triton.jit
def bailey_scale_kernel(
    x_re_ptr, x_im_ptr,     # (rows*m0*M,) fp16 input (logical (rows, m0, M))
    y_re_ptr, y_im_ptr,     # (rows*m0*M,) fp16 output ((rows, m0, M) or (rows, M, m0))
    tw_re_ptr, tw_im_ptr,   # (m0, M) fp16
    m0, M,
    BLOCK_M0: tl.constexpr,
    BLOCK_M: tl.constexpr,
    STORE_T: tl.constexpr,
):
    """Elementwise complex multiply by bt[n1, kM] over the (rows, m0, M) view.
    fp32 arithmetic, fp16 result. STORE_T=True fuses with a transpose to
    produce (rows, M, m0).

    Grid: (cdiv(m0, BLOCK_M0), cdiv(M, BLOCK_M), rows).

    TODO: implement.
    """
    pid_m0 = tl.program_id(0)
    pid_m = tl.program_id(1)
    row = tl.program_id(2)
 
    offs_m0 = pid_m0 * BLOCK_M0 + tl.arange(0, BLOCK_M0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
 
    mask = (offs_m0[:, None] < m0) & (offs_m[None, :] < M)
 
    # Input index: row * m0 * M + n1 * M + kM
    in_idx = row * m0 * M + offs_m0[:, None] * M + offs_m[None, :]
    x_re = tl.load(x_re_ptr + in_idx, mask=mask, other=0.0).to(tl.float32)
    x_im = tl.load(x_im_ptr + in_idx, mask=mask, other=0.0).to(tl.float32)
 
    tw_re = tl.load(tw_re_ptr + offs_m0[:, None] * M + offs_m[None, :], mask=mask, other=0.0).to(tl.float32)
    tw_im = tl.load(tw_im_ptr + offs_m0[:, None] * M + offs_m[None, :], mask=mask, other=0.0).to(tl.float32)
 
    y_re = (x_re * tw_re - x_im * tw_im).to(tl.float16)
    y_im = (x_re * tw_im + x_im * tw_re).to(tl.float16)
 
    if STORE_T:
        # Fuse with T2: output layout (rows, M, m0) -> index: row * M * m0 + kM * m0 + n1
        out_idx = row * m0 * M + offs_m[None, :] * m0 + offs_m0[:, None]
    else:
        out_idx = row * m0 * M + offs_m0[:, None] * M + offs_m[None, :]
 
    tl.store(y_re_ptr + out_idx, y_re, mask=mask)
    tl.store(y_im_ptr + out_idx, y_im, mask=mask)


# =============================================================================
# Thin launch wrappers -- GIVEN, do not edit
# =============================================================================

def _transpose(in_re, in_im, out_re, out_im, B, R, C):
    """Logical (B, R, C) -> (B, C, R) transpose, paired re/im."""
    grid = (triton.cdiv(R, TRANSPOSE_BLOCK), triton.cdiv(C, TRANSPOSE_BLOCK), B)
    transpose_kernel[grid](
        in_re, in_im, out_re, out_im, R, C,
        BLOCK_R=TRANSPOSE_BLOCK, BLOCK_C=TRANSPOSE_BLOCK,
    )


def _fft_chunk(in_re, in_im, out_re, out_im, rows, m, plan, M=1, store_t=False):
    """Length-m FFT over `rows` contiguous (rows, m) signals.

    M / store_t control the output layout:
      store_t=False, M=1: natural (rows, m) row-major (F6 leaf path)
      store_t=True,  M>1: transposed (rows//M, m, M) (F7 fused FFT-m0+T3)
    """
    if m == 256:
        f4_plan = plan['f4_plan']
        f4_kernel_L2[(triton.cdiv(rows, F4_L2_BLOCK_B),)](
            in_re.view(rows, 256), in_im.view(rows, 256),
            out_re.view(rows, 256), out_im.view(rows, 256),
            f4_plan['F_re'], f4_plan['F_im'],
            f4_plan['tw_re'], f4_plan['tw_im'],
            rows, M,
            BLOCK_B=F4_L2_BLOCK_B, STAGE_STOP=f4_plan['L'], STORE_T=store_t,
            num_warps=4, num_stages=1,
        )
    else:
        M_re, M_im = plan['dft_mats'][m]
        dft_kernel[(triton.cdiv(rows, DFT_BLOCK_B),)](
            in_re.view(rows, m), in_im.view(rows, m),
            out_re.view(rows, m), out_im.view(rows, m),
            M_re, M_im, rows, M,
            R=m, BLOCK_B=DFT_BLOCK_B, STORE_T=store_t,
        )


def _scale(in_re, in_im, out_re, out_im, rows, m0, M, twr, twi, store_t=False):
    """Bailey scale over logical (rows, m0, M)."""
    grid = (triton.cdiv(m0, SCALE_BLOCK), triton.cdiv(M, SCALE_BLOCK), rows)
    bailey_scale_kernel[grid](
        in_re, in_im, out_re, out_im, twr, twi,
        m0, M, BLOCK_M0=SCALE_BLOCK, BLOCK_M=SCALE_BLOCK, STORE_T=store_t,
    )


def _lookup_tw(plan, m0, M, N_i):
    """Find the precomputed Bailey twiddle table for (m0, M, N_i) in plan['tw']."""
    for (a, b, n, tr, ti) in plan['tw']:
        if a == m0 and b == M and n == N_i:
            return tr, ti
    raise KeyError(f"no twiddle table for (m0={m0}, M={M}, N={N_i})")


# =============================================================================
# F3 pipeline: 4-step Bailey six-step (T1 -> F2-A -> T2 -> F2-B)
# =============================================================================

def f3_launch(in_re, in_im, out_re, out_im, mid_re, mid_im, plan, B):
    """Run the 4-step F3 pipeline. Buffer ping-pong: in -> mid -> out -> mid
    -> out. The Bailey twiddle fuses into F2-A (BAILEY_EPILOGUE=True), and
    the would-be T3 is absorbed by F2-B (STRIDED_STORE=True).

    Steps:
      1. T1 (transpose): x[b, n2, n1] -> A[b, n1, n2]
      2. F2-A:           length-N2 FFT over (B*N1) signals with Bailey epilogue
      3. T2 (transpose): Z[b, n1, k2] -> Z'[b, k2, n1]
      4. F2-B:           length-N1 FFT over (B*N2) signals with strided store

    TODO: implement.
    """
    N1 = plan['N1']
    N2 = plan['N2']
    N = plan['N']
    LOG2_N1 = plan['LOG2_N1']
    LOG2_N2 = plan['LOG2_N2']
 
    # Step 1: T1 -- (B, N2, N1) -> (B, N1, N2)
    # in_re is flat (B*N), view as (B, N2, N1) -> transpose to (B, N1, N2) -> mid
    _transpose(in_re, in_im, mid_re, mid_im, B, N2, N1)
 
    # Step 2: F2-A -- length-N2 FFT over (B*N1) rows, with Bailey epilogue
    # mid_re: (B, N1, N2) -> flat, pid = b*N1 + n1
    # BAILEY_EPILOGUE=True, OUTER_DIM=N1, bt shape (N1, N2)
    f2_kernel[(B * N1,)](
        mid_re, mid_im, out_re, out_im,
        plan['tw_re_n2'], plan['tw_im_n2'],
        plan['perm_n2'],
        plan['bt_re'], plan['bt_im'],
        N1, 0,
        N=N2, LOG2_N=LOG2_N2,
        BAILEY_EPILOGUE=True, STRIDED_STORE=False,
    )
 
    # Step 3: T2 -- (B, N1, N2) -> (B, N2, N1)
    _transpose(out_re, out_im, mid_re, mid_im, B, N1, N2)
 
    # Step 4: F2-B -- length-N1 FFT over (B*N2) rows, strided store
    # STRIDED_STORE=True: output stored as (B, N1, N2) with stride N2
    f2_kernel[(B * N2,)](
        mid_re, mid_im, out_re, out_im,
        plan['tw_re_n1'], plan['tw_im_n1'],
        plan['perm_n1'],
        plan['tw_re_n1'], plan['tw_im_n1'],  # sentinel
        N2, N,
        N=N1, LOG2_N=LOG2_N1,
        BAILEY_EPILOGUE=False, STRIDED_STORE=True,
    )


# =============================================================================
# F5 pipeline: 6-step Bailey at N1=N2=256 with F4 as inner FFT
# =============================================================================

def f5_launch(in_re, in_im, b0_re, b0_im, b1_re, b1_im, b2_re, b2_im, plan, B):
    """Run the 6-step F5 pipeline at N = 65536 = 256 * 256.

    Buffer ping-pong: in -> b0 -> b1 -> b0 -> b1 -> b2 -> b0 (final).
    The Bailey twiddle is NOT fused into F4 (F4 stays unmodified), so this is
    6 launches; F7 generalizes the fusion idea recursively.

    Steps:
      1. T1:    x[b, n2, n1] -> A[b, n1, n2]
      2. FFT-A: length-256 FFT along last axis -> Y[b, n1, k2]
      3. Scale: Z[b, n1, k2] = Y[b, n1, k2] * bt[n1, k2]
      4. T2:    Z[b, n1, k2] -> Z'[b, k2, n1]
      5. FFT-B: length-256 FFT along last axis -> V[b, k2, k1]
      6. T3:    V[b, k2, k1] -> X[b, k1, k2]   (final in b0)

    TODO: implement.
    """
    N1 = plan['N1']  # 256
    N2 = plan['N2']  # 256
    N = plan['N']    # 65536
    f4_plan = plan['f4_plan']
 
    # T1: (B, N2, N1) -> (B, N1, N2), in -> b0
    _transpose(in_re, in_im, b0_re, b0_im, B, N2, N1)
 
    # F4-A: length-256 FFT over B*N1 rows, b0 -> b1
    rows_a = B * N1
    f4_kernel_L2[(triton.cdiv(rows_a, F4_L2_BLOCK_B),)](
        b0_re.view(rows_a, 256), b0_im.view(rows_a, 256),
        b1_re.view(rows_a, 256), b1_im.view(rows_a, 256),
        f4_plan['F_re'], f4_plan['F_im'],
        f4_plan['tw_re'], f4_plan['tw_im'],
        rows_a, 1,
        BLOCK_B=F4_L2_BLOCK_B, STAGE_STOP=f4_plan['L'], STORE_T=False,
        num_warps=4, num_stages=1,
    )
 
    # Scale: multiply by bailey cross-twiddle bt[n1, k2], b1 -> b0
    # Logical shape: (B, N1, N2)
    _scale(b1_re, b1_im, b0_re, b0_im, B, N1, N2, plan['bt_re'], plan['bt_im'], store_t=False)
 
    # T2: (B, N1, N2) -> (B, N2, N1), b0 -> b1
    _transpose(b0_re, b0_im, b1_re, b1_im, B, N1, N2)
 
    # F4-B: length-256 FFT over B*N2 rows, b1 -> b2
    rows_b = B * N2
    f4_kernel_L2[(triton.cdiv(rows_b, F4_L2_BLOCK_B),)](
        b1_re.view(rows_b, 256), b1_im.view(rows_b, 256),
        b2_re.view(rows_b, 256), b2_im.view(rows_b, 256),
        f4_plan['F_re'], f4_plan['F_im'],
        f4_plan['tw_re'], f4_plan['tw_im'],
        rows_b, 1,
        BLOCK_B=F4_L2_BLOCK_B, STAGE_STOP=f4_plan['L'], STORE_T=False,
        num_warps=4, num_stages=1,
    )
 
    # T3: (B, N2, N1) -> (B, N1, N2), b2 -> b0 (final)
    _transpose(b2_re, b2_im, b0_re, b0_im, B, N2, N1)


# =============================================================================
# F6 / F7 recursion
# =============================================================================
# Per level i with chunks = [m_0, m_1, ..., m_{p-1}], M = prod(chunks[1:]):
#   T1 :       (rows, M, m_0) -> (rows, m_0, M)
#   recurse:   length-M FFT over (rows*m_0, M)
#   Scale :    y *= w_{N_i}^{n_1 k_M}            (n_1 = the m_0 digit)
#   T2 :       (rows, m_0, M) -> (rows, M, m_0)
#   FFT-m_0 :  length-m_0 FFT over (rows*M, m_0)
#   T3 :       (rows, M, m_0) -> (rows, m_0, M)   [F6 only; F7 fuses]

def _f6_rec(cur_re, cur_im, rows, chunks, plan, cyc):
    """Recursive 2-factor Bailey split. Leaf (len(chunks)==1) is one
    _fft_chunk call; non-leaf is the 6-step pipeline above.

    Returns the (re, im) cycler-managed buffers holding the (rows, prod(chunks))
    FFT result.

    TODO: implement.
    """
    m0 = chunks[0]
    M = math.prod(chunks[1:]) if len(chunks) > 1 else 1
    N_i = m0 * M
 
    if len(chunks) == 1:
        # Leaf: one FFT of length m0 over `rows` signals
        out_re, out_im = cyc.next()
        _fft_chunk(cur_re, cur_im, out_re, out_im, rows, m0, plan, M=1, store_t=False)
        return out_re, out_im
 
    # Non-leaf: T1 -> recurse -> Scale -> T2 -> FFT-m0 -> T3
    # Input: (rows, M, m0) -- wait, cur is (rows*N_i,) flat, logically (rows, m0, M)?
    # According to ALGORITHMS.md: input shape is (rows, M, m0), T1 transposes to (rows, m0, M)
 
    # T1: (rows, M, m0) -> (rows, m0, M)
    t1_re, t1_im = cyc.next()
    _transpose(cur_re, cur_im, t1_re, t1_im, rows, M, m0)
 
    # recurse: length-M FFT over (rows*m0) signals
    # Input to recursion: (rows*m0, M) flat
    rec_out_re, rec_out_im = _f6_rec(t1_re, t1_im, rows * m0, chunks[1:], plan, cyc)
 
    # Scale: multiply by bailey twiddle bt[n1, kM] shape (m0, M)
    # Logical shape of rec_out: (rows, m0, M)
    twr, twi = _lookup_tw(plan, m0, M, N_i)
    sc_re, sc_im = cyc.next()
    _scale(rec_out_re, rec_out_im, sc_re, sc_im, rows, m0, M, twr, twi, store_t=False)
 
    # T2: (rows, m0, M) -> (rows, M, m0)
    t2_re, t2_im = cyc.next()
    _transpose(sc_re, sc_im, t2_re, t2_im, rows, m0, M)
 
    # FFT-m0: length-m0 FFT over (rows*M) signals
    fft_re, fft_im = cyc.next()
    _fft_chunk(t2_re, t2_im, fft_re, fft_im, rows * M, m0, plan, M=1, store_t=False)
 
    # T3: (rows, M, m0) -> (rows, m0, M)
    t3_re, t3_im = cyc.next()
    _transpose(fft_re, fft_im, t3_re, t3_im, rows, M, m0)
 
    return t3_re, t3_im


def _f7_rec(cur_re, cur_im, rows, chunks, plan, cyc):
    """Same recursion as _f6_rec but with Scale+T2 fused (store_t=True on
    bailey_scale_kernel) and FFT-m_0+T3 fused (store_t=True, M=M on the inner
    FFT kernel). Output should be bitwise-equal to _f6_rec.

    TODO: implement.
    """
    m0 = chunks[0]
    M = math.prod(chunks[1:]) if len(chunks) > 1 else 1
    N_i = m0 * M
 
    if len(chunks) == 1:
        out_re, out_im = cyc.next()
        _fft_chunk(cur_re, cur_im, out_re, out_im, rows, m0, plan, M=1, store_t=False)
        return out_re, out_im
 
    # T1: (rows, M, m0) -> (rows, m0, M)
    t1_re, t1_im = cyc.next()
    _transpose(cur_re, cur_im, t1_re, t1_im, rows, M, m0)
 
    # recurse
    rec_out_re, rec_out_im = _f7_rec(t1_re, t1_im, rows * m0, chunks[1:], plan, cyc)
 
    # Scale + T2 fused (store_t=True): (rows, m0, M) -> scaled (rows, M, m0)
    twr, twi = _lookup_tw(plan, m0, M, N_i)
    sc_t2_re, sc_t2_im = cyc.next()
    _scale(rec_out_re, rec_out_im, sc_t2_re, sc_t2_im, rows, m0, M, twr, twi, store_t=True)
 
    # FFT-m0 + T3 fused (store_t=True, M=M): (rows, M, m0) -> (rows, m0, M)
    out_re, out_im = cyc.next()
    _fft_chunk(sc_t2_re, sc_t2_im, out_re, out_im, rows * M, m0, plan, M=M, store_t=True)
 
    return out_re, out_im