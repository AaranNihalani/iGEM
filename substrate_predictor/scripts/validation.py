"""External validation of the MMP-9 model on the Kridel 2001 measurements, with leakage control."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
from spred import data, validation as V
from spred.model import fit_model
from spred.evaluation import hamming_clusters

OUT = Path(__file__).resolve().parents[1] / "results" / "validation"; OUT.mkdir(parents=True, exist_ok=True)
kr = V.load_kridel()
pos = data.trainable_positives()

# ---- leakage audit: training windows (any protease) that coincide with a Kridel peptide window
leak = []
for pid, seqs in pos.items():
    for s in seqs:
        for r in kr.itertuples():
            if V.matches(r.window, s) and sum(c != "-" for c in r.window) >= 7:
                leak.append((pid, s, r.peptide))
leak_df = pd.DataFrame(leak, columns=["protease", "training_window", "kridel_peptide"])
leak_df.to_csv(OUT / "leakage_audit.csv", index=False)
print("training windows coinciding with a Kridel peptide (>=7 shared positions):", len(leak_df))
print(leak_df.groupby("protease").size().to_dict() if len(leak_df) else "none")

# ---- retrain MMP-9 with all coinciding windows removed from every set (positives and cross-negatives)
drop = set(leak_df.training_window)
clean = {p: [s for s in v if s not in drop] for p, v in pos.items()}
fr = data.training_frame("MMP9", clean)
g = hamming_clusters(fr.sequence.tolist())
model = fit_model("MMP9", fr, clean["MMP9"], groups=g)

rows = []
for r in kr.itertuples():
    rows.append({"peptide": r.peptide, "sequence": r.sequence, "window": r.window, "n_positions": sum(c != "-" for c in r.window),
                 "kcat_km": r.mmp9_kcat_km, "status": r.mmp9_status,
                 "lr_logit": V.partial_logit(model, r.window), "pssm": V.partial_pssm(model, r.window)})
df = pd.DataFrame(rows)
# standardise both scores (z over the 12 peptides) and average -> simple consensus for this small set
for c in ("lr_logit", "pssm"): df[c + "_z"] = (df[c] - df[c].mean()) / df[c].std()
df["consensus_z"] = (df.lr_logit_z + df.pssm_z) / 2
df.to_csv(OUT / "kridel_scores.csv", index=False)

meas = df[df.status == "measured"].copy()
y11 = np.log10(meas.kcat_km.values)
y12 = np.r_[y11, y11.min() - 1.0]                   # C11 (no activity detected) ranked below every measured peptide
res = []
for name in ("lr_logit", "pssm", "consensus_z"):
    for label, x, y in (("n=11 measured", meas[name].values, y11), ("n=12 incl. C11 as lowest", df[name].values[[*meas.index, df.index[df.status != "measured"][0]]], y12)):
        rho, p, lo, hi = V.spearman_with_permutation(x, y)
        res.append({"score": name, "set": label, "spearman": rho, "perm_p": p, "ci_lo": lo, "ci_hi": hi})
r = pd.DataFrame(res); r.to_csv(OUT / "kridel_spearman.csv", index=False)
print(r.round(3).to_string(index=False))
print(df[["peptide", "window", "kcat_km", "lr_logit", "pssm", "consensus_z"]].round(2).to_string(index=False))

# ---- matched mutant pairs from Kridel 2001 (same peptide background, single substitution): direction and size of effect
pairs = [("A11", "A11m1", "T7R at P1 (Thr to Arg)"), ("A11", "A11m2", "L8A at P1' (Leu to Ala)")]
rows = []
byname = kr.set_index("peptide")
for a, b, note in pairs:
    ra, rb = byname.loc[a], byname.loc[b]
    la, lb = V.partial_logit(model, ra.window), V.partial_logit(model, rb.window)
    pa, pb = V.partial_pssm(model, ra.window), V.partial_pssm(model, rb.window)
    rows.append({"parent": a, "mutant": b, "change": note, "measured_fold": rb.mmp9_kcat_km / ra.mmp9_kcat_km,
                 "measured_log2_fold": float(np.log2(rb.mmp9_kcat_km / ra.mmp9_kcat_km)),
                 "model_delta_logit": lb - la, "model_delta_logit_bits": (lb - la) / np.log(2), "model_delta_pssm_bits": pb - pa})
pd.DataFrame(rows).to_csv(OUT / "kridel_mutant_pairs.csv", index=False)
print(pd.DataFrame(rows).round(3).to_string(index=False))
