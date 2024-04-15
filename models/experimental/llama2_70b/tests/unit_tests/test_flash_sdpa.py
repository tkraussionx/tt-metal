import torch
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)


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

    out = torch.zeros_like(q)

    for b in range(num_batches):
        for h in range(num_heads):
            for q_c in range(num_chunks):
                q_chunk = q[b, h, q_c * chunk_size : (q_c + 1) * chunk_size]
                mij, mij1 = torch.full((chunk_size, 1), -1000), torch.full((chunk_size, 1), -1000)
                lij, lij1 = torch.full((chunk_size, 1), 0), torch.full((chunk_size, 1), 0)
                chunk_output = torch.zeros_like(q_chunk)
                for k_c in range(num_chunks):
                    # MQA: only 1 kv head
                    # 7
                    k_chunk = k[b, 0, k_c * chunk_size : (k_c + 1) * chunk_size]
                    v_chunk = v[b, 0, k_c * chunk_size : (k_c + 1) * chunk_size]
                    attn_mask_chunk = attn_mask[
                        b, h, q_c * chunk_size : (q_c + 1) * chunk_size, k_c * chunk_size : (k_c + 1) * chunk_size
                    ]

                    # 8
                    scores = torch.matmul(q_chunk, k_chunk.transpose(-1, -2))
                    scores = scores / (q_chunk.size(-1) ** 0.5)
                    scores = scores + attn_mask_chunk

                    # 9
                    rowmax = torch.max(scores, dim=-1, keepdim=True)[0]
                    mij = torch.maximum(mij1, rowmax)
                    pij = torch.exp(scores - mij)
                    row_sum = torch.sum(pij, dim=-1, keepdim=True)

                    lij = torch.exp(mij1 - mij) * lij1 + row_sum

                    # 10
                    chunk_output = chunk_output * torch.exp(mij1 - mij) + torch.matmul(pij, v_chunk)

                    lij1 = lij
                    mij1 = mij

                # 12
                chunk_output = chunk_output / lij
                out[b, h, q_c * chunk_size : (q_c + 1) * chunk_size] = chunk_output

    return out


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
