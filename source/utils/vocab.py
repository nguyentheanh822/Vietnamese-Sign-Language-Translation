"""utils/vocab.py — Vocabulary management cho Gloss va Translation."""
import re
from pathlib import Path
from collections import Counter

# Special tokens
PAD_TOKEN   = "<pad>"
BOS_TOKEN   = "<bos>"
EOS_TOKEN   = "<eos>"
UNK_TOKEN   = "<unk>"
BLANK_TOKEN = "<blank>"   # CTC blank

SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN, BLANK_TOKEN]


class Vocab:
    """Vocabulary cho mot tap token (gloss hoac translation)."""

    def __init__(self, tokens=None):
        self.token2idx = {}
        self.idx2token = []
        for tok in SPECIAL_TOKENS:
            self._add(tok)
        if tokens:
            for tok in tokens:
                self._add(tok)

    def _add(self, token):
        if token not in self.token2idx:
            self.token2idx[token] = len(self.idx2token)
            self.idx2token.append(token)

    def __len__(self):
        return len(self.idx2token)

    @property
    def pad_idx(self):  return self.token2idx[PAD_TOKEN]
    @property
    def bos_idx(self):  return self.token2idx[BOS_TOKEN]
    @property
    def eos_idx(self):  return self.token2idx[EOS_TOKEN]
    @property
    def unk_idx(self):  return self.token2idx[UNK_TOKEN]
    @property
    def blank_idx(self): return self.token2idx[BLANK_TOKEN]

    def encode(self, tokens):
        """List token -> list int"""
        return [self.token2idx.get(t, self.unk_idx) for t in tokens]

    def decode(self, indices, skip_special=True):
        """List int -> list token"""
        result = []
        for i in indices:
            tok = self.idx2token[i] if i < len(self.idx2token) else UNK_TOKEN
            if skip_special and tok in SPECIAL_TOKENS:
                if tok == EOS_TOKEN:
                    break
                continue
            result.append(tok)
        return result

    def save(self, path):
        Path(path).write_text(
            "\n".join(self.idx2token), encoding="utf-8"
        )

    @classmethod
    def load(cls, path):
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        # Bo cac dong special token neu da co
        tokens = [l for l in lines if l and l not in SPECIAL_TOKENS]
        return cls(tokens)


def build_translation_vocab(dataset_json_path, min_freq=1):
    """
    Tao vocabulary cho ban dich tieng Viet tu dataset.json.
    Tokenize don gian: tach theo dau cach + lowercase + giu dau (unicode).
    """
    import json
    with open(dataset_json_path, encoding="utf-8") as f:
        dataset = json.load(f)

    counter = Counter()
    for entry in dataset:
        trans = entry.get("translation", "")
        if trans:
            tokens = tokenize_vi(trans)
            counter.update(tokens)

    tokens_sorted = [t for t, c in counter.most_common() if c >= min_freq]
    return Vocab(tokens_sorted)


def tokenize_vi(text):
    """
    Tokenize tieng Viet don gian: lowercase + split theo space/punctuation.
    Giu dau tieng Viet (unicode normalization).
    """
    text = text.strip().lower()
    # Tach punctuation khoi tu
    text = re.sub(r"([.,!?;:\"'()\[\]])", r" \1 ", text)
    tokens = text.split()
    return [t for t in tokens if t]
