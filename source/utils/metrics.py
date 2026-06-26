"""
utils/metrics.py
----------------
Tinh WER (Word Error Rate) cho CSLR va BLEU-4 cho SLT.
"""

import re
from collections import Counter
import math


# ── WER ──────────────────────────────────────────────────────────────────────
def wer(hypothesis: list, reference: list) -> float:
    """
    Word Error Rate: tinh tren danh sach token.
    WER = (S + D + I) / N
    """
    H, R = len(hypothesis), len(reference)
    if R == 0:
        return 0.0 if H == 0 else 1.0

    # Dynamic programming
    dp = list(range(H + 1))
    for i in range(1, R + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, H + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return dp[H] / R


def batch_wer(hypotheses: list, references: list) -> float:
    """Mean WER tren tap danh sach."""
    if not hypotheses:
        return 0.0
    scores = [wer(h, r) for h, r in zip(hypotheses, references)]
    return sum(scores) / len(scores)


# ── BLEU ─────────────────────────────────────────────────────────────────────
def _ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def bleu_n(hypothesis: list, reference: list, n: int) -> tuple:
    """Tinh precision n-gram (clip)."""
    hyp_ngrams = Counter(_ngrams(hypothesis, n))
    ref_ngrams = Counter(_ngrams(reference, n))
    clipped = {ng: min(cnt, ref_ngrams[ng]) for ng, cnt in hyp_ngrams.items()}
    return sum(clipped.values()), max(len(hypothesis) - n + 1, 0)


def bleu4(hypotheses: list, references: list) -> float:
    """
    Corpus-level BLEU-4.
    hypotheses, references: list of list of str tokens.
    """
    total_match = [0] * 4
    total_count = [0] * 4
    hyp_len_total = 0
    ref_len_total = 0

    for hyp, ref in zip(hypotheses, references):
        hyp_len_total += len(hyp)
        ref_len_total += len(ref)
        for n in range(1, 5):
            m, c = bleu_n(hyp, ref, n)
            total_match[n-1] += m
            total_count[n-1] += c

    # Log precision
    log_bp = 0.0
    if hyp_len_total < ref_len_total:
        log_bp = 1.0 - ref_len_total / max(hyp_len_total, 1)

    log_bleu = log_bp
    for n in range(4):
        if total_count[n] == 0 or total_match[n] == 0:
            return 0.0
        log_bleu += 0.25 * math.log(total_match[n] / total_count[n])

    return math.exp(log_bleu) * 100.0   # return as percentage


# ── NEW METRICS (chrF, ROUGE-L, BERTScore) ───────────────────────────────────
def chrf_score(hypotheses: list, references: list) -> float:
    import sacrebleu
    # sacrebleu expects list of strings
    hyp_strs = [" ".join(h) for h in hypotheses]
    ref_strs = [[" ".join(r) for r in references]] # list of reference streams
    if not hyp_strs: return 0.0
    chrf = sacrebleu.corpus_chrf(hyp_strs, ref_strs)
    return chrf.score

def rouge_l_score(hypotheses: list, references: list) -> float:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    scores = []
    for h, r in zip(hypotheses, references):
        h_str = " ".join(h)
        r_str = " ".join(r)
        scores.append(scorer.score(r_str, h_str)['rougeL'].fmeasure)
    if not scores: return 0.0
    return (sum(scores) / len(scores)) * 100.0

def bert_score_metric(hypotheses: list, references: list) -> float:
    import bert_score
    import torch
    hyp_strs = [" ".join(h) for h in hypotheses]
    ref_strs = [" ".join(r) for r in references]
    if not hyp_strs: return 0.0
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # We use a smaller multilingual model for speed and compatibility, or basic bert
    # using 'bert-base-multilingual-cased' works for Vietnamese
    P, R, F1 = bert_score.score(hyp_strs, ref_strs, lang="vi", verbose=False, device=device)
    return F1.mean().item() * 100.0


# ── Helper: decode token IDs to strings ─────────────────────────────────────
def ids_to_tokens(ids_batch, vocab, skip_special=True):
    """Batch of list[int] -> list of list[str]"""
    return [vocab.decode(ids, skip_special=skip_special) for ids in ids_batch]

