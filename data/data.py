from datasets import load_dataset
from data.tokenizer import build_vocab, TokenizedDataset
from torch.utils.data import DataLoader

def get_wikitext_loaders(batch_size=16, max_length=64, vocab_size=2000, return_train_split=False):
    """Load the Wikitext dataset and create DataLoaders."""

    # Load dataset from Hugging Face
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    # Build vocabulary from training split
    vocab = build_vocab(dataset, vocabSize=vocab_size)

    # Create tokenized datasets for all splits
    train_dataset = TokenizedDataset(dataset["train"], vocab, max_length)
    val_dataset   = TokenizedDataset(dataset["validation"], vocab, max_length)
    test_dataset  = TokenizedDataset(dataset["test"], vocab, max_length)

    # Wrap in PyTorch DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size)

    # Return loaders and vocabulary, and optionally the training split (for weighted loss)
    if return_train_split:
        return train_loader, val_loader, test_loader, vocab, dataset["train"]
    return train_loader, val_loader, test_loader, vocab