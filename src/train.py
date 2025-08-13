import numpy as np
import random
import sys
import torch
from torch import nn
from torch.optim import AdamW
from collections import Counter
from src.transformer import MaskedLanguageModel
from src.masking import mask_tokens
from data.data import get_wikitext_loaders
from src.viz import print_batch, save_attention_overlays

sys.path.append("/content/mini-transformer")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def evaluate(model, data_loader, loss_fn, device, mask_token_id, vocab_size):
    """Evaluate the model on the given validation data."""
    model.eval()
    total_loss = 0.0 # Initialize for accumulation for validation set
    total_correct = 0
    total_masked = 0
    with torch.no_grad():
        for batch in data_loader:
            masked_inputs, labels, _ = mask_tokens(
                batch["input_ids"].to(device),
                mask_token_id=mask_token_id,
                vocab_size=vocab_size,
            )
            outputs = model(masked_inputs, return_attention=False)

            # Calculate loss
            loss = loss_fn(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            total_loss += loss.item()

            # Calculate accuracy on masked positions
            preds = outputs.argmax(dim=-1)
            m = (labels != -100)
            total_correct += (preds[m] == labels[m]).sum().item()
            total_masked += m.sum().item()
    avg_loss = total_loss / len(data_loader)
    avg_acc  = (total_correct / total_masked) if total_masked > 0 else 0.0
    return avg_loss, avg_acc

def final_eval(model, data_loader, loss_fn, vocab, device):
    """Final evaluation on the test set."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    kept_correct = kept_total = 0
    mask_correct = mask_total = 0
    rand_correct = rand_total = 0

    for batch in data_loader:
        x0 = batch["input_ids"].to(device)

        # Randomly mask tokens in the input batch
        masked_inputs, labels, _ = mask_tokens(x0, mask_token_id=vocab["[MASK]"], vocab_size=len(vocab))
        masked_inputs = masked_inputs.to(device); labels = labels.to(device)

        mask_any = (labels != -100).any()
        if not mask_any:
            continue  # skip empty-mask batch
        
        # Forward pass, skip attention scores for simplicity
        logits = model(masked_inputs, return_attention=False)
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        if torch.isnan(loss):
            continue  # safety

        total_loss += loss.item(); n_batches += 1

        # Accuracy calculation on MLM-supervised positions
        preds = logits.argmax(-1)
        kept_mask   = (labels != -100) & (masked_inputs == x0)
        mask_mask   = (labels != -100) & (masked_inputs == vocab["[MASK]"])
        random_mask = (labels != -100) & ~(kept_mask | mask_mask)

        kept_correct += (preds[kept_mask] == labels[kept_mask]).sum().item(); kept_total   += kept_mask.sum().item()
        mask_correct += (preds[mask_mask] == labels[mask_mask]).sum().item(); mask_total   += mask_mask.sum().item()
        rand_correct += (preds[random_mask] == labels[random_mask]).sum().item(); rand_total   += random_mask.sum().item()

    return {
        "val_loss": (total_loss / n_batches) if n_batches else float("nan"),
        "acc_mask":   (mask_correct / mask_total) if mask_total else 0.0,
        "acc_kept":   (kept_correct / kept_total) if kept_total else 0.0,
        "acc_random": (rand_correct / rand_total) if rand_total else 0.0,
    }



def compute_token_frequencies(train_split, vocab):
    """Compute token frequencies in the training split."""
    counter = Counter()
    for ex in train_split:
        ids = [vocab.get(tok, vocab["[UNK]"]) for tok in ex["text"].split()]
        counter.update(ids)
    return counter

def compute_loss_weights(counter, vocab_size, smoothing=1.0, mask_token_id=None):
    """Compute loss weights based on token frequencies."""
    freqs = torch.zeros(vocab_size)
    for idx, freq in counter.items():
        freqs[idx] = freq

    weights = 1.0 / (freqs + smoothing)

    if mask_token_id is not None:
        weights[mask_token_id] = 0.0  # Ignore [MASK] token in loss

    weights = weights / weights.sum() * vocab_size  # Normalize
    return weights



# Load the dataset and create data loaders
train_loader, val_loader, test_loader, vocab, train_split = \
    get_wikitext_loaders(return_train_split=True)


# Initialize the model
model = MaskedLanguageModel(num_layers=4,
                            embed_dim=256,
                            vocab_size=len(vocab),
                            num_heads=8,
                            hidden_dim=512,
                            max_length=64).to(device)

# Initialize the optimizer and loss function
optimizer = AdamW(model.parameters(), lr=1e-4)

counter = compute_token_frequencies(train_split, vocab) # Compute token frequencies
loss_weights = compute_loss_weights(counter, vocab_size=len(vocab), mask_token_id=vocab["[MASK]"]).to(device)
loss_func = nn.CrossEntropyLoss(weight=loss_weights, ignore_index=-100) # Use weighted loss function, ignore -100 labels

# Initialize lists to store loss and accuracy history
loss_history = []
accuracy_history = []

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

best_val_loss = float('inf')
patience = 6
stalled = 0
best_state = None 

epochs = 100  # Number of epochs to train

for epoch in range(epochs): # Training Loop
    model.train()
    total_loss = 0.0  # Initialize for accumulation per epoch
    total_correct = 0
    total_masked = 0
    print(f"Epoch {epoch+1}/{epochs}")

    for step, batch in enumerate(train_loader):

        # Randomly mask tokens in the input batch
        masked_inputs, labels, maskArr = mask_tokens(batch['input_ids'],
                                                   mask_token_id=vocab["[MASK]"],
                                                   vocab_size = len(vocab))
        masked_inputs = masked_inputs.to(device)
        labels = labels.to(device)

        # Forward pass, skip attention scores for simplicity
        outputs = model(masked_inputs, return_attention=False)

        predictions = torch.argmax(outputs, dim=-1)

        if step == 0: # Print the first batch for debugging purposes and visualization
            print("Printing batch...")
            print_batch(masked_inputs, labels, predictions, vocab)

        loss = loss_func(outputs.view(-1, outputs.size(-1)), labels.view(-1))

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Accuracy calculation on MLM-supervised positions
        correct = (predictions == labels) & (labels != -100)
        total_correct += correct.sum().item()
        total_masked += (labels != -100).sum().item()

    # Average loss and accuracy for the epoch
    avg_loss = total_loss / len(train_loader)
    accuracy = total_correct / total_masked if total_masked > 0 else 0.0
    loss_history.append(avg_loss)
    accuracy_history.append(accuracy)

    # Evaluate on validation set
    val_loss, val_acc = evaluate(model, val_loader, loss_func, device,
                                 mask_token_id=vocab["[MASK]"], vocab_size=len(vocab))
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    
    # Print training and validation metrics
    print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.4f}|Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # Early stopping logic
    if val_loss + 1e-4 < best_val_loss:  # min_delta
        best_val_loss = val_loss
        stalled = 0
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    else:
        stalled += 1
        if stalled >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

if best_state is not None:
    torch.save(best_state, "best_model.pth")
    print("Best model saved to best_model.pth")

# Final evaluation on the test set
results = final_eval(model, test_loader, loss_func, vocab, device)
print("Final evaluation:", results)

# Plot the attention overlays for the first few test examples
save_attention_overlays(model, test_loader, vocab, device,
                        out_dir="attn_overlays", max_images=12, seed=42)