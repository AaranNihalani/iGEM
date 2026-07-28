"""
PSSM-based protease cleavage scorer with BLOSUM62 pseudocounts.

Each protease gets a Position-Specific Scoring Matrix built from its aligned
P4-P4' positive windows.  Pseudocounts use the BLOSUM62 substitution matrix
(Henikoff & Henikoff 1992) to share information between chemically similar
amino acids — critical when n_positives is small (e.g. LasB n=35).

Score(seq) = sum_{pos=0..7} log2( (f_aa_pos + beta * q_aa_pos) / bg_aa )

where:
  f_aa_pos  = observed frequency of aa at pos
  q_aa_pos  = BLOSUM62-weighted expectation (substitution-averaged background)
  beta      = pseudocount weight (default 5; ~equivalent to 5 pseudo-observations)
  bg_aa     = Swiss-Prot amino acid background frequency

The score is then converted to a sigmoid probability for comparison with the
logistic-regression model.  Both scorers are reported; agreement between LR and
PSSM is the confidence signal.
"""

import logging
from pathlib import Path
import numpy as np
from typing import Optional

log = logging.getLogger(__name__)

STANDARD_AAS = list('ACDEFGHIKLMNPQRSTVWY')
AA_IDX = {aa: i for i, aa in enumerate(STANDARD_AAS)}
WINDOW_LEN = 8

# Swiss-Prot background frequencies (UniProtKB/Swiss-Prot 2023)
SWISSPROT_BG = {
    'A': 0.0825, 'R': 0.0553, 'N': 0.0406, 'D': 0.0546, 'C': 0.0138,
    'Q': 0.0393, 'E': 0.0674, 'G': 0.0708, 'H': 0.0227, 'I': 0.0590,
    'L': 0.0965, 'K': 0.0581, 'M': 0.0241, 'F': 0.0385, 'P': 0.0476,
    'S': 0.0658, 'T': 0.0534, 'W': 0.0109, 'Y': 0.0292, 'V': 0.0686,
}
BG = np.array([SWISSPROT_BG[aa] for aa in STANDARD_AAS], dtype=float)

# BLOSUM62 matrix — rows/cols ordered by STANDARD_AAS
# Source: NCBI BLOSUM62
_BLOSUM62_RAW = {
    'A': { 'A':4,'R':-1,'N':-2,'D':-2,'C':0,'Q':-1,'E':-1,'G':0,'H':-2,'I':-1,'L':-1,'K':-1,'M':-1,'F':-2,'P':-1,'S':1,'T':0,'W':-3,'Y':-2,'V':0},
    'R': { 'A':-1,'R':5,'N':0,'D':-2,'C':-3,'Q':1,'E':0,'G':-2,'H':0,'I':-3,'L':-2,'K':2,'M':-1,'F':-3,'P':-2,'S':-1,'T':-1,'W':-3,'Y':-2,'V':-3},
    'N': { 'A':-2,'R':0,'N':6,'D':1,'C':-3,'Q':0,'E':0,'G':0,'H':1,'I':-3,'L':-3,'K':0,'M':-2,'F':-3,'P':-2,'S':1,'T':0,'W':-4,'Y':-2,'V':-3},
    'D': { 'A':-2,'R':-2,'N':1,'D':6,'C':-3,'Q':0,'E':2,'G':-1,'H':-1,'I':-3,'L':-4,'K':-1,'M':-3,'F':-3,'P':-1,'S':0,'T':-1,'W':-4,'Y':-3,'V':-3},
    'C': { 'A':0,'R':-3,'N':-3,'D':-3,'C':9,'Q':-3,'E':-4,'G':-3,'H':-3,'I':-1,'L':-1,'K':-3,'M':-1,'F':-2,'P':-3,'S':-1,'T':-1,'W':-2,'Y':-2,'V':-1},
    'Q': { 'A':-1,'R':1,'N':0,'D':0,'C':-3,'Q':5,'E':2,'G':-2,'H':0,'I':-3,'L':-2,'K':1,'M':0,'F':-3,'P':-1,'S':0,'T':-1,'W':-2,'Y':-1,'V':-2},
    'E': { 'A':-1,'R':0,'N':0,'D':2,'C':-4,'Q':2,'E':5,'G':-2,'H':0,'I':-3,'L':-3,'K':1,'M':-2,'F':-3,'P':-1,'S':0,'T':-1,'W':-3,'Y':-2,'V':-2},
    'G': { 'A':0,'R':-2,'N':0,'D':-1,'C':-3,'Q':-2,'E':-2,'G':6,'H':-2,'I':-4,'L':-4,'K':-2,'M':-3,'F':-3,'P':-2,'S':0,'T':-2,'W':-2,'Y':-3,'V':-3},
    'H': { 'A':-2,'R':0,'N':1,'D':-1,'C':-3,'Q':0,'E':0,'G':-2,'H':8,'I':-3,'L':-3,'K':-1,'M':-2,'F':-1,'P':-2,'S':-1,'T':-2,'W':-2,'Y':2,'V':-3},
    'I': { 'A':-1,'R':-3,'N':-3,'D':-3,'C':-1,'Q':-3,'E':-3,'G':-4,'H':-3,'I':4,'L':2,'K':-3,'M':1,'F':0,'P':-3,'S':-2,'T':-1,'W':-3,'Y':-1,'V':3},
    'L': { 'A':-1,'R':-2,'N':-3,'D':-4,'C':-1,'Q':-2,'E':-3,'G':-4,'H':-3,'I':2,'L':4,'K':-2,'M':2,'F':0,'P':-3,'S':-2,'T':-1,'W':-2,'Y':-1,'V':1},
    'K': { 'A':-1,'R':2,'N':0,'D':-1,'C':-3,'Q':1,'E':1,'G':-2,'H':-1,'I':-3,'L':-2,'K':5,'M':-1,'F':-3,'P':-1,'S':0,'T':-1,'W':-3,'Y':-2,'V':-2},
    'M': { 'A':-1,'R':-1,'N':-2,'D':-3,'C':-1,'Q':0,'E':-2,'G':-3,'H':-2,'I':1,'L':2,'K':-1,'M':5,'F':0,'P':-2,'S':-1,'T':-1,'W':-1,'Y':-1,'V':1},
    'F': { 'A':-2,'R':-3,'N':-3,'D':-3,'C':-2,'Q':-3,'E':-3,'G':-3,'H':-1,'I':0,'L':0,'K':-3,'M':0,'F':6,'P':-4,'S':-2,'T':-2,'W':1,'Y':3,'V':-1},
    'P': { 'A':-1,'R':-2,'N':-2,'D':-1,'C':-3,'Q':-1,'E':-1,'G':-2,'H':-2,'I':-3,'L':-3,'K':-1,'M':-2,'F':-4,'P':7,'S':-1,'T':-1,'W':-4,'Y':-3,'V':-2},
    'S': { 'A':1,'R':-1,'N':1,'D':0,'C':-1,'Q':0,'E':0,'G':0,'H':-1,'I':-2,'L':-2,'K':0,'M':-1,'F':-2,'P':-1,'S':4,'T':1,'W':-3,'Y':-2,'V':-2},
    'T': { 'A':0,'R':-1,'N':0,'D':-1,'C':-1,'Q':-1,'E':-1,'G':-2,'H':-2,'I':-1,'L':-1,'K':-1,'M':-1,'F':-2,'P':-1,'S':1,'T':5,'W':-2,'Y':-2,'V':0},
    'W': { 'A':-3,'R':-3,'N':-4,'D':-4,'C':-2,'Q':-2,'E':-3,'G':-2,'H':-2,'I':-3,'L':-2,'K':-3,'M':-1,'F':1,'P':-4,'S':-3,'T':-2,'W':11,'Y':2,'V':-3},
    'Y': { 'A':-2,'R':-2,'N':-2,'D':-3,'C':-2,'Q':-1,'E':-2,'G':-3,'H':2,'I':-1,'L':-1,'K':-2,'M':-1,'F':3,'P':-3,'S':-2,'T':-2,'W':2,'Y':7,'V':-1},
    'V': { 'A':0,'R':-3,'N':-3,'D':-3,'C':-1,'Q':-2,'E':-2,'G':-3,'H':-3,'I':3,'L':1,'K':-2,'M':1,'F':-1,'P':-2,'S':-2,'T':0,'W':-3,'Y':-1,'V':4},
}

# Build BLOSUM62 as numpy array [20 x 20]
BLOSUM62 = np.zeros((20, 20), dtype=float)
for i, a in enumerate(STANDARD_AAS):
    for j, b in enumerate(STANDARD_AAS):
        BLOSUM62[i, j] = _BLOSUM62_RAW[a][b]

# Convert to probability-like weights (row-normalised positive part)
_B62_pos = np.maximum(BLOSUM62, 0)
_B62_row_sum = _B62_pos.sum(axis=1, keepdims=True)
_B62_row_sum[_B62_row_sum == 0] = 1.0
BLOSUM62_NORM = _B62_pos / _B62_row_sum   # shape [20,20], rows sum to 1


class PSSM:
    """
    Position-Specific Scoring Matrix for protease cleavage prediction.

    Attributes
    ----------
    log_odds : np.ndarray, shape (8, 20)
        log2 odds score for each (position, amino acid).
    n_positives : int
        Number of positive sequences used to build the PSSM.
    protease : str
        Identifier for which protease this PSSM represents.
    """

    def __init__(self, protease: str = ''):
        self.protease = protease
        self.log_odds: Optional[np.ndarray] = None
        self.n_positives = 0
        self._score_mean = 0.0
        self._score_std = 1.0

    def fit(self, positives: list[str], beta: float = 5.0) -> 'PSSM':
        """
        Build PSSM from aligned P4-P4' positive windows.

        Parameters
        ----------
        positives : list of 8-char uppercase AA strings.
        beta      : pseudocount weight (≈ number of pseudo-observations).
                    Higher = more smoothing toward BLOSUM62 expected frequencies.
        """
        n = len(positives)
        if n == 0:
            raise ValueError("No positives provided")
        self.n_positives = n

        # Raw counts: [8 positions x 20 AAs]
        counts = np.zeros((WINDOW_LEN, 20), dtype=float)
        valid = 0
        for seq in positives:
            if len(seq) != WINDOW_LEN:
                continue
            ok = True
            for aa in seq:
                if aa not in AA_IDX:
                    ok = False
                    break
            if not ok:
                continue
            for pos, aa in enumerate(seq):
                counts[pos, AA_IDX[aa]] += 1
            valid += 1

        if valid == 0:
            raise ValueError("No valid 8-mer positives after filtering")

        # Convert to frequencies
        freq = counts / counts.sum(axis=1, keepdims=True).clip(min=1)

        # BLOSUM62 pseudocounts: q_aa_pos = sum_aa' freq_aa' * BLOSUM62_NORM[aa', aa]
        # Gives the BLOSUM62-weighted expectation for each position
        q = freq @ BLOSUM62_NORM   # [8 x 20]

        # Pseudo-enriched frequency
        enriched = (freq * valid + beta * q) / (valid + beta)

        # Log-odds vs Swiss-Prot background
        bg = BG.copy()
        bg = bg / bg.sum()
        with np.errstate(divide='ignore', invalid='ignore'):
            self.log_odds = np.log2(np.where(enriched > 0, enriched / bg, 1e-9))

        return self

    def score(self, seq: str) -> float:
        """Sum of positional log-odds for a single 8-mer (raw score)."""
        if self.log_odds is None:
            raise RuntimeError("PSSM not fitted")
        if len(seq) != WINDOW_LEN:
            return float('-inf')
        s = 0.0
        for pos, aa in enumerate(seq):
            idx = AA_IDX.get(aa.upper())
            if idx is None:
                return float('-inf')
            s += self.log_odds[pos, idx]
        return s

    def score_batch(self, seqs: list[str]) -> np.ndarray:
        """Vectorised scoring of many 8-mers; invalid seqs get -inf."""
        if self.log_odds is None:
            raise RuntimeError("PSSM not fitted")
        n = len(seqs)
        # Encode to integer index matrix [n, 8]; -1 for invalid
        idx_mat = np.full((n, WINDOW_LEN), -1, dtype=np.int8)
        for i, seq in enumerate(seqs):
            if len(seq) != WINDOW_LEN:
                continue
            ok = True
            for pos, aa in enumerate(seq.upper()):
                j = AA_IDX.get(aa, -1)
                if j == -1:
                    ok = False
                    break
                idx_mat[i, pos] = j
            if not ok:
                idx_mat[i, :] = -1

        valid = (idx_mat >= 0).all(axis=1)
        scores = np.full(n, float('-inf'))
        if valid.any():
            pos_idx = np.arange(WINDOW_LEN)
            # Gather log-odds: log_odds[pos, aa] for each (row, pos)
            gathered = self.log_odds[pos_idx, idx_mat[valid]]  # [n_valid, 8]
            scores[valid] = gathered.sum(axis=1)
        return scores

    def calibrate(self, all_seqs: list[str]) -> 'PSSM':
        """
        Compute mean and std of scores on a held-out or training set
        so that scores can be z-normalised for cross-model comparison.
        """
        scores = [self.score(s) for s in all_seqs]
        self.mu = float(np.mean(scores))
        self.sigma = float(np.std(scores)) or 1.0
        return self

    def score_zscore(self, seq: str) -> float:
        """Z-normalised PSSM score (mean 0, std 1 over calibration set)."""
        return (self.score(seq) - self.mu) / self.sigma

    def score_percentile(self, seq: str, reference_scores: np.ndarray) -> float:
        """Fraction of reference scores below this sequence's score."""
        raw = self.score(seq)
        return float((reference_scores < raw).mean())

    def top_aa_at_position(self, pos: int, n: int = 5) -> list[tuple[str, float]]:
        """Return top-n amino acids by log-odds at a given position (0-indexed)."""
        if self.log_odds is None:
            return []
        lo = self.log_odds[pos]
        order = np.argsort(lo)[::-1]
        return [(STANDARD_AAS[i], float(lo[i])) for i in order[:n]]

    def save(self, path: Path) -> None:
        np.save(path, self.log_odds)

    def load(self, path: Path, protease: str = '') -> 'PSSM':
        self.log_odds = np.load(path)
        self.protease = protease
        return self


def build_pssm_from_positives(positives: list[str], protease: str,
                               beta: float = 5.0) -> PSSM:
    pssm = PSSM(protease=protease)
    pssm.fit(positives, beta=beta)
    return pssm
