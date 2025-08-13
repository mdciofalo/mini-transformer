import collections
import torch
from torch.utils.data import Dataset

def build_vocab(dataset, vocab_size=2000):
    """Build a vocabulary from the dataset."""
    # Count word frequencies in the training split
    counter = collections.Counter()

    for example in dataset["train"]:
        words = example["text"].split()
        counter.update(words)

    # Create a vocabulary mapping from word to index
    vocab = {"[PAD]": 0, "[MASK]": 1, "[UNK]": 2}
    for idx, (word, _) in enumerate(counter.most_common(vocab_size - len(vocab)), start=len(vocab)):
        vocab[word] = idx

    return vocab

def tokenize(text, vocab, max_length=64):
    """Tokenize a text string into a list of token IDs."""
    tokens = text.split()
    ids = [vocab.get(token, vocab["[UNK]"]) for token in tokens]
    ids = ids[:max_length]
    # Pad to max_length
    ids += [vocab["[PAD]"]] * (max_length - len(ids))
    return ids

class TokenizedDataset(Dataset):
    """Dataset that tokenizes text examples into input IDs."""
    def __init__(self, split, vocab, max_length=64):
        self.examples = []
        for example in split:
            token_ids = tokenize(example["text"], vocab, max_length)
            self.examples.append(torch.tensor(token_ids, dtype=torch.long))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return {"input_ids": self.examples[idx]}
