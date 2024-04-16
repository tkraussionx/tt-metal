import torch
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)
import tt_lib
import ttnn
from models.utility_functions import torch2tt_tensor, tt2torch_tensor


def sdpa(q, k, v, attn_mask):
    scores = torch.matmul(q, k.transpose(-1, -2))
    scores = scores / (q.size(-1) ** 0.5)
    scores = scores + attn_mask
    attn = torch.nn.functional.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)
    return out


def fa2(q, k, v, attn_mask):
    chunk_size = 256
    num_batches = q.size(0)
    num_heads = q.size(1)
    seq_len = q.size(2)
    num_chunks = seq_len // chunk_size

    scale = 1.0 / (q.size(-1) ** 0.5)

    out = torch.zeros_like(q)

    for b in range(num_batches):
        for h in range(num_heads):
            for q_c in range(num_chunks):
                # Fetch Q chunk
                q_chunk = q[b, h, q_c * chunk_size : (q_c + 1) * chunk_size]

                # Setup variables for the Q chunk
                cur_m, prev_m = torch.full((chunk_size, 1), -1000), torch.full((chunk_size, 1), -1000)
                cur_sum, prev_sum = torch.full((chunk_size, 1), 0), torch.full((chunk_size, 1), 0)
                chunk_output = torch.zeros_like(q_chunk)

                for k_c in range(num_chunks):
                    # MQA: only 1 kv head

                    # Fetch K, V, attn_mask chunks
                    k_chunk = k[b, 0, k_c * chunk_size : (k_c + 1) * chunk_size]
                    v_chunk = v[b, 0, k_c * chunk_size : (k_c + 1) * chunk_size]
                    attn_mask_chunk = attn_mask[
                        b, h, q_c * chunk_size : (q_c + 1) * chunk_size, k_c * chunk_size : (k_c + 1) * chunk_size
                    ]

                    scores = torch.matmul(q_chunk, k_chunk.transpose(-1, -2))
                    scores = scores * scale
                    scores = scores + attn_mask_chunk

                    # 9
                    rowmax = torch.max(scores, dim=-1, keepdim=True)[0]
                    cur_m = torch.maximum(prev_m, rowmax)
                    pij = torch.exp(scores - cur_m)
                    row_sum = torch.sum(pij, dim=-1, keepdim=True)

                    exp_max_diff = torch.exp(prev_m - cur_m)

                    cur_sum = exp_max_diff * prev_sum + row_sum

                    # 10
                    chunk_output = chunk_output * exp_max_diff + torch.matmul(pij, v_chunk)

                    prev_sum = cur_sum
                    prev_m = cur_m

                # 12
                chunk_output = chunk_output / cur_sum
                out[b, h, q_c * chunk_size : (q_c + 1) * chunk_size] = chunk_output

    return out


def fa2_fake(q, k, v, attn_mask):
    chunk_size = 256
    num_batches = q.size(0)
    num_heads = q.size(1)
    seq_len = q.size(2)
    num_chunks = seq_len // chunk_size

    scale = 1.0 / (q.size(-1) ** 0.5)

    out = torch.zeros_like(q)

    for b in range(num_batches):
        for h in range(num_heads):
            for q_c in range(num_chunks):
                # Fetch Q chunk
                q_chunk = q[b, h, q_c * chunk_size : (q_c + 1) * chunk_size]

                # Setup variables for the Q chunk
                """Initialize with good values for the first row"""
                cur_m, prev_m = torch.full((chunk_size, 1), -1000), torch.full((chunk_size, 1), -1000)
                cur_sum, prev_sum = torch.full((chunk_size, 1), 0), torch.full((chunk_size, 1), 0)
                chunk_output = torch.zeros_like(q_chunk)

                for k_c in range(num_chunks):
                    # MQA: only 1 kv head

                    # Fetch K, V, attn_mask chunks
                    k_chunk = k[b, 0, k_c * chunk_size : (k_c + 1) * chunk_size]
                    v_chunk = v[b, 0, k_c * chunk_size : (k_c + 1) * chunk_size]
                    # attn_mask_chunk: cb_mask
                    attn_mask_chunk = attn_mask[
                        b, h, q_c * chunk_size : (q_c + 1) * chunk_size, k_c * chunk_size : (k_c + 1) * chunk_size
                    ]

                    scores = torch.matmul(q_chunk, k_chunk.transpose(-1, -2))

                    """broadcast_mult"""
                    scores = scores * scale  # scale: cb_scale (single tile column vector)
                    """eltwise add inplace"""
                    # scores = scores + attn_mask_chunk

                    # 9
                    """reduce max (c)"""
                    # rowmax = torch.max(scores, dim=-1, keepdim=True)[0] # cb_cur_max (S_chunk_t tiles)
                    """reduce max (c)"""
                    # cur_m = torch.maximum(prev_m, rowmax)
                    """broadcast sub"""
                    """eltwise_unary exp"""
                    # pij = torch.exp(scores - cur_m)
                    """reduce sum (c)"""
                    # row_sum = torch.sum(pij, dim=-1, keepdim=True)
                    """col-vector sub"""
                    """eltwise_unary exp"""
                    # exp_max_diff = torch.exp(prev_m - cur_m)
                    """eltwise mult"""
                    """eltwise add"""
                    # cur_sum = exp_max_diff * prev_sum + row_sum

                    # # 10
                    """broadcast_mult inplace"""
                    """matmul"""
                    """eltwise add inplace"""
                    # chunk_output = chunk_output * exp_max_diff + torch.matmul(pij, v_chunk)
                    chunk_output += torch.matmul(scores, v_chunk)
                    ## ONLY KEEP LAST CHUNK OUTPUT
                    # chunk_output = torch.matmul(scores, v_chunk)
                    """copy tiles"""
                    # prev_sum = cur_sum
                    # prev_m = cur_m

                # 12
                """reciprocal"""
                """broadcast mult"""
                # chunk_output = chunk_output / cur_sum
                """copy tiles"""
                out[b, h, q_c * chunk_size : (q_c + 1) * chunk_size] = chunk_output

    return out


def tt_fa2(device, q, k, v, attn_mask):
    tt_q = torch2tt_tensor(q, device)
    tt_k = torch2tt_tensor(k.transpose(-1, -2), device)
    tt_v = torch2tt_tensor(v, device)
    tt_attn_mask = torch2tt_tensor(attn_mask, device)

    tt_out = tt_lib.operations.primary.transformers.scaled_dot_product_attention(
        tt_q, tt_k, tt_v, tt_attn_mask, is_causal=True
    )

    return tt2torch_tensor(tt_out)


def run_test_sdpa_tt(device):
    b = 1
    nh = 8
    nkv = 1
    s = 2048
    d = 128

    Q = torch.randn(b, nh, s, d)
    K = torch.randn(b, nkv, s, d)
    V = torch.randn(b, nkv, s, d)
    attn_mask = torch.full((s, s), torch.finfo(torch.float32).min)
    attn_mask = torch.triu(attn_mask, diagonal=1).expand(b, nh, -1, -1)

    # Print shapes of all inputs along with input names
    print(f"Q: {Q.shape}")
    print(f"K: {K.shape}")
    print(f"V: {V.shape}")
    print(f"attn_mask: {attn_mask.shape}")

    # is_causal must be false if we specify an attention_mask
    gt = fa2_fake(
        Q,
        K,
        V,
        attn_mask,
    )
    mine = tt_fa2(device, Q, K, V, attn_mask)
    out_pass, out_pcc = comp_pcc(gt, mine, 0.99)
    print(f"python vs pytorch: {out_pcc}")

    row_tiles = s // 32
    col_tiles = d // 32
    for batch in range(b):
        for head in range(nh):
            for row_tile in range(row_tiles):
                for col_tile in range(col_tiles):
                    gt_tile = gt[batch, head, row_tile * 32 : (row_tile + 1) * 32, col_tile * 32 : (col_tile + 1) * 32]
                    mine_tile = mine[
                        batch, head, row_tile * 32 : (row_tile + 1) * 32, col_tile * 32 : (col_tile + 1) * 32
                    ]
                    # Print MSE if these tiles
                    mse = torch.nn.functional.mse_loss(gt_tile, mine_tile)
                    if mse > 0.6:
                        print(f"Tile {row_tile}, {col_tile}")
                        print(f"MSE: {mse}")

    assert out_pass


def run_test_sdpa_python():
    b = 1
    nh = 8
    nkv = 1
    s = 2048
    d = 128

    Q = torch.randn(b, nh, s, d)
    K = torch.randn(b, nkv, s, d)
    V = torch.randn(b, nkv, s, d)
    attn_mask = torch.full((s, s), torch.finfo(torch.float32).min)
    attn_mask = torch.triu(attn_mask, diagonal=1).expand(b, nh, -1, -1)

    # Print shapes of all inputs along with input names
    print(f"Q: {Q.shape}")
    print(f"K: {K.shape}")
    print(f"V: {V.shape}")
    print(f"attn_mask: {attn_mask.shape}")

    # is_causal must be false if we specify an attention_mask
    gt = torch.nn.functional.scaled_dot_product_attention(Q, K, V, attn_mask, is_causal=False)

    mine = sdpa(Q, K, V, attn_mask)

    out_pass, out_pcc = comp_pcc(gt, mine, 0.99)
    print(f"python vs pytorch: {out_pcc}")
    assert out_pass

    fa = fa2(Q, K, V, attn_mask)
    out_pass, out_pcc = comp_pcc(gt, fa, 0.99)
    print(f"fa2 vs pytorch: {out_pcc}")
    assert out_pass


def test_sdpa_python():
    run_test_sdpa_python()


def test_sdpa_tt(device):
    run_test_sdpa_tt(device)
