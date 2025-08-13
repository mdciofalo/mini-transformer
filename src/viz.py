from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import os
import torch
import numpy as np
from src.masking import mask_tokens

def print_batch(input_ids, labels, predictions, vocab, max_examples=3):
    """Print `True vs Pred` for masked positions in up to `max_examples` items."""
    
    id_to_word = {v: k for k, v in vocab.items()}
    pad_id = vocab["[PAD]"]

    B, L = input_ids.shape
    for i in range(min(B, max_examples)):
        inp  = input_ids[i].detach().cpu().tolist()
        gold = labels[i].detach().cpu().tolist()
        pred = predictions[i].detach().cpu().tolist()

        trunc = inp.index(pad_id) if pad_id in inp else L
        gold, pred = gold[:trunc], pred[:trunc]

        for j, lbl in enumerate(gold):
            if lbl != -100:  # masked position
                true_tok = id_to_word.get(lbl, "[UNK]")
                pred_tok = id_to_word.get(pred[j], "[UNK]")
                print("True vs Pred:", true_tok, pred_tok)


def render_attention_overlay(tokens, weights, masked_idx=None, out_path="attn.png",
                             width=1600, pad=30, base_font_size=28,
                             circle_base=36, blur=12,
                             text_color=(0,0,0,255), stroke=2,
                             heat_rgb=(70,120,255)):
    """Render attention weights as an overlay on the input tokens."""

    # Tokens must be strings
    tokens = [str(t) for t in tokens]
    w = np.asarray(weights, dtype=np.float32)
    w = np.clip(w, 0, None)
    w = (w - w.min()) / (w.max() - w.min() + 1e-8) if w.max() > 0 else w

    img = Image.new("RGBA", (width, base_font_size + pad*2 + 10), (255,255,255,255))
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", base_font_size)
    except Exception:
        font = ImageFont.load_default()

    # Measure positions
    draw = ImageDraw.Draw(img)
    x, y = pad, pad
    centers = []
    for i, tok in enumerate(tokens):
        disp = tok if i == len(tokens)-1 else tok + " "
        tok_w = draw.textlength(disp, font=font)
        centers.append((x + tok_w/2, y + base_font_size/2 + 2))
        x += tok_w

    # Heat overlay first
    heat = Image.new("RGBA", img.size, (0,0,0,0))
    hdraw = ImageDraw.Draw(heat)
    for (cx, cy), a in zip(centers, w):
        if a <= 0: 
            continue
        r = int(circle_base * (0.3 + 0.7*np.sqrt(a)))
        cx, cy = int(cx), int(cy)
        bbox = [cx-r, cy-r, cx+r, cy+r]
        hdraw.ellipse(bbox, fill=(heat_rgb[0], heat_rgb[1], heat_rgb[2], int(110*a)))  # lower alpha
    heat = heat.filter(ImageFilter.GaussianBlur(blur))
    img = Image.alpha_composite(img, heat)

    # Draw text on TOP (with white stroke for readability)
    draw = ImageDraw.Draw(img)
    x = pad
    for i, tok in enumerate(tokens):
        disp = tok if i == len(tokens)-1 else tok + " "
        draw.text((x, y), disp, font=font, fill=text_color,
                  stroke_width=stroke, stroke_fill=(255,255,255,255))
        x += draw.textlength(disp, font=font)

    # Optional: ring the masked token
    if masked_idx is not None and 0 <= masked_idx < len(tokens):
        cx, cy = centers[masked_idx]
        r = int(circle_base*0.9)
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], outline=(240,80,80,255), width=3)

    img.convert("RGB").save(out_path)
    return out_path

def save_attention_overlays(model, data_loader, vocab, device,
                            out_dir="attn_overlays", max_images=12, seed=123):
    os.makedirs(out_dir, exist_ok=True)
    """Save attention overlays for masked tokens in the test set."""

    id2tok = {v:k for k,v in vocab.items()}
    PAD, MASK = vocab["[PAD]"], vocab["[MASK]"]

    # Make eval deterministic (so masks are the same each run)
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    model.eval()
    saved = 0
    with torch.no_grad():
        for b_idx, batch in enumerate(data_loader):
            x0 = batch["input_ids"].to(device)

            # Dynamic masking at eval time (like train)
            x, y, _ = mask_tokens(x0, mask_token_id=MASK, vocab_size=len(vocab))
            logits, attn_scores = model(x, return_attention=True)
            preds = logits.argmax(-1)

            B, L = x.shape
            for i in range(B):
                if saved >= max_images: break

                seq = x[i].tolist()
                trunc = seq.index(PAD) if PAD in seq else L
                tokens = [id2tok.get(t, "[UNK]") for t in seq[:trunc]]

                # Pick a visible [MASK] position to highlight
                mpos_list = ((y[i, :trunc] != -100) & (x[i, :trunc] == MASK)).nonzero(as_tuple=True)[0]
                if len(mpos_list) == 0:
                    continue
                mpos = int(mpos_list[0].item())

                # Last layer, average heads -> weights for that masked position
                if isinstance(attn_scores, list) or attn_scores.dim() == 5:
                    # [layers, batch, heads, seq, seq]
                    attn = attn_scores[-1][i].mean(dim=0)    # [seq, seq]
                else:
                    attn = attn_scores[i].mean(dim=0)

                weights = attn[mpos, :trunc].detach().cpu().numpy()

                true_tok = id2tok.get(int(y[i, mpos].item()), "[UNK]")
                pred_tok = id2tok.get(int(preds[i, mpos].item()), "[UNK]")

                # Prints sentence and top-5 predictions

                sentence = " ".join(tokens)
                probs = torch.softmax(logits[i, mpos], dim=-1)
                pred_id = int(preds[i, mpos].item())
                pred_prob = probs[pred_id].item()
                topk_vals, topk_idx = torch.topk(probs, k=5)
                top5 = [(id2tok.get(int(j.item()), "[UNK]"), float(v.item())) for v, j in zip(topk_vals, topk_idx)]

                print("Sentence:", sentence)
                print(f"Masked pos {mpos} | TRUE: '{true_tok}' | PRED: '{pred_tok}' ({pred_prob:.1%})")
                print("Top-5 :", ", ".join(f"{t} {p:.1%}" for t, p in top5))

                # Save attention overlay
                out_path = os.path.join(
                    out_dir, f"test_b{b_idx}_i{i}_pos{mpos}_true-{true_tok}_pred-{pred_tok}.png"
                )
                render_attention_overlay(tokens, weights, masked_idx=mpos, out_path=out_path)
                saved += 1

            if saved >= max_images: break

    # Restore RNG state (optional)
    torch.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)

    print(f"Saved {saved} attention overlays to: {out_dir}")