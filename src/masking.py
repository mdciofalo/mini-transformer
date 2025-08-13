import torch

def mask_tokens(
    inputs,
    mask_token_id,
    vocab_size,
    pad_token_id=0,
    unk_token_id=2,
    mask_prob=0.15,
):
    x = inputs.clone()
    labels = inputs.clone()
    device = x.device

    # Choose which positions to mask 
    r_mask = torch.rand_like(x, dtype=torch.float)
    mask_arr = (r_mask < mask_prob) & (x != pad_token_id) & (x != unk_token_id)
    labels[~mask_arr] = -100  # compute loss only on masked positions

    # Choose category among masked tokens (80/10/10) 
    r_cat = torch.rand_like(x, dtype=torch.float)

    mask_mask = mask_arr & (r_cat < 0.8)                      # 80% -> [MASK]
    rand_mask = mask_arr & (r_cat >= 0.8) & (r_cat < 0.9)     # 10% -> random
    keep_mask = mask_arr & (r_cat >= 0.9)                     # 10% -> keep

    x[mask_mask] = mask_token_id
    # Random token (avoid special ids with an index table; no resampling)
    if rand_mask.any():
        avoid = {pad_token_id, unk_token_id, mask_token_id}
        allowed = torch.tensor(
            [i for i in range(vocab_size) if i not in avoid],
            device=device
        )
        idx = torch.randint(0, allowed.numel(), size=x.shape, device=device)
        rand_tokens = allowed[idx]
        x[rand_mask] = rand_tokens[rand_mask]

    # Keep_mask: do nothing (leave original token)

    return x, labels, mask_arr

