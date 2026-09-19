"""Generate LaTeX tables and number macros from results/ so that no reported value is typed by hand."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
R = ROOT / "results"
T = Path(__file__).resolve().parent / "tables"
T.mkdir(exist_ok=True)
ORDER = ["MMP2", "MMP8", "MMP9", "MMP14", "NE", "LASB", "FECAL"]
NAME = {"MMP2": "MMP-2", "MMP8": "MMP-8", "MMP9": "MMP-9", "MMP14": "MMP-14", "NE": "Neutrophil elastase", "LASB": "LasB", "FECAL": "Faecal serine proteases"}
WORD = {"MMP2": "MMPtwo", "MMP8": "MMPeight", "MMP9": "MMPnine", "MMP14": "MMPfourteen", "NE": "NE", "LASB": "LasB", "FECAL": "Faecal"}
f3 = lambda x: f"{x:.3f}"
f2 = lambda x: f"{x:.2f}"


def write(name, text):
    (T / f"{name}.tex").write_text(text)


def ci(v, lo, hi):
    return f"{v:.3f} \\,[{lo:.3f}, {hi:.3f}]"


cv = pd.read_csv(R / "cv_summary.csv").set_index("protease")
# ---------------------------------------------------------------- number macros
macros = []
for p in ORDER:
    r = cv.loc[p]
    w = WORD[p]
    macros += [(f"nPos{w}", f"{int(r.n_pos):,}".replace(",", "{,}")), (f"nNeg{w}", f"{int(r.n_neg):,}".replace(",", "{,}")),
               (f"aucLR{w}", f3(r.lr_auroc)), (f"aucPSSM{w}", f3(r.pssm_auroc)), (f"aucCons{w}", f3(r.consensus_auroc)),
               (f"aucLRci{w}", ci(r.lr_auroc, r.lr_auroc_lo, r.lr_auroc_hi)), (f"aucConsci{w}", ci(r.consensus_auroc, r.consensus_auroc_lo, r.consensus_auroc_hi))]
kr = pd.read_csv(R / "validation" / "kridel_spearman.csv")
for _, r in kr.iterrows():
    tag = ("NEleven" if r.set.startswith("n=11") else "NTwelve") + {"lr_logit": "LR", "pssm": "PSSM", "consensus_z": "Cons"}[r.score]
    macros += [(f"rho{tag}", f2(r.spearman)), (f"rhoP{tag}", f3(r.perm_p)), (f"rhoLo{tag}", f2(r.ci_lo)), (f"rhoHi{tag}", f2(r.ci_hi))]
mp = pd.read_csv(R / "validation" / "kridel_mutant_pairs.csv").set_index("mutant")
macros += [("mutOneMeasured", f2(mp.loc["A11m1", "measured_log2_fold"])), ("mutOneLR", f2(mp.loc["A11m1", "model_delta_logit_bits"])), ("mutOnePSSM", f2(mp.loc["A11m1", "model_delta_pssm_bits"])),
           ("mutTwoMeasured", f2(mp.loc["A11m2", "measured_log2_fold"])), ("mutTwoLR", f2(mp.loc["A11m2", "model_delta_logit_bits"])), ("mutTwoPSSM", f2(mp.loc["A11m2", "model_delta_pssm_bits"]))]
lk = pd.read_csv(R / "methodology" / "random_vs_cluster_cv.csv").set_index("protease")
macros += [("maxLeakGap", f3(float((lk.auroc_random_cv - lk.auroc_cluster_cv).abs().max())))]
ls = pd.read_csv(R / "methodology" / "label_shuffle_control.csv")
macros += [("shuffleMin", f2(ls.auroc_label_shuffled_mean.min())), ("shuffleMax", f2(ls.auroc_label_shuffled_mean.max()))]
cs = pd.read_csv(R / "methodology" / "vhse_subspace_coverage_summary.csv").set_index("protease")
macros += [("subspaceMin", f"{100 * cs.r2_weighted.min():.0f}"), ("subspaceMax", f"{100 * cs.r2_weighted.max():.0f}")]
from spred.model import load_all as _la
from spred.tables import AMINO_ACIDS as _AA
_m9 = _la()["MMP9"].log_odds
def _rk(pos, aa):
    i = _AA.index(aa); return 1 + int((_m9[pos] > _m9[pos, i]).sum()), float(_m9[pos, i])
for tag, (pos, aa) in {"PThreePro": (1, "P"), "PTwoArg": (2, "R"), "POneThr": (3, "T"), "POneSer": (3, "S"), "POnePLeu": (4, "L"), "POnePIle": (4, "I"), "PTwoPSer": (5, "S"), "PTwoPThr": (5, "T"), "POneGly": (3, "G"), "POneLys": (3, "K"), "POneArg": (3, "R")}.items():
    r_, v_ = _rk(pos, aa); macros += [(f"pssmRank{tag}", str(r_)), (f"pssmLO{tag}", f"{v_:+.2f}")]
(T.parent / "numbers.tex").write_text("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros) + "\n")

# ---------------------------------------------------------------- T1 main results
rows = []
for p in ORDER:
    r = cv.loc[p]
    vals = [r.lr_auroc, r.pssm_auroc, r.consensus_auroc]
    cis = [(r.lr_auroc_lo, r.lr_auroc_hi), (r.pssm_auroc_lo, r.pssm_auroc_hi), (r.consensus_auroc_lo, r.consensus_auroc_hi)]
    cells = []
    for v, (lo_, hi_) in zip(vals, cis):
        t = ci(v, lo_, hi_)
        cells.append(f"\\textbf{{{t}}}" if v >= max(vals) - 1e-12 else t)
    rows.append(f"{NAME[p]} & {int(r.n_pos):,} & {int(lk.loc[p, 'n_pos_clusters']):,} & " + " & ".join(cells) + " \\\\")
write("main_results", "{\\footnotesize\\begin{tabular}{@{}lrrccc@{}}\\toprule\nProtease & Positives & Positive clusters & LR AUROC & PSSM AUROC & Consensus AUROC \\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}}\n")
rows = []
for p in ORDER:
    r = cv.loc[p]
    rows.append(f"{NAME[p]} & {f3(r.lr_pr_auc)} & {f3(r.lr_mcc)} & {f3(r.lr_brier)} & {f3(r.lr_ece)} & {r.chosen_C.strip('[]')} \\\\")
write("secondary_metrics", "\\begin{tabular}{@{}lccccl@{}}\\toprule\nProtease & PR-AUC & MCC & Brier & ECE & $C$ chosen per outer fold \\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n")

# ---------------------------------------------------------------- T2 model comparison
mc = pd.read_csv(R / "methodology" / "model_comparison.csv")
models = ["prior", "composition-LR", "one-hot LR", "VHSE LR", "PSSM", "one-hot GBM", "VHSE GBM"]
rows = []
for p in ORDER:
    vals = {m: mc[(mc.protease == p) & (mc.model == m)].auroc.iloc[0] for m in models}
    best = max(v for m, v in vals.items())
    cells = [f"\\textbf{{{v:.3f}}}" if abs(v - best) < 1e-9 else f"{v:.3f}" for v in [vals[m] for m in models]]
    rows.append(f"{NAME[p]} & " + " & ".join(cells) + " \\\\")
write("model_comparison", "\\begin{tabular}{@{}lccccccc@{}}\\toprule\nProtease & Prior & Composition & One-hot LR & VHSE LR & PSSM & One-hot GBM & VHSE GBM \\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n")
d = pd.read_csv(R / "methodology" / "paired_differences_vs_vhse_lr.csv")
rows = []
for p in ORDER:
    cells = []
    for m in ["one-hot LR", "PSSM", "one-hot GBM", "VHSE GBM"]:
        r = d[(d.protease == p) & (d.vs == m)].iloc[0]
        cells.append(f"{-r.delta_auroc:+.3f}" + ("$^{*}$" if r.significant else ""))
    rows.append(f"{NAME[p]} & " + " & ".join(cells) + " \\\\")
write("paired_differences", "\\begin{tabular}{@{}lcccc@{}}\\toprule\nProtease & One-hot LR & PSSM & One-hot GBM & VHSE GBM \\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n")

# ---------------------------------------------------------------- T4 negative design
a = pd.read_csv(R / "methodology" / "negative_design_ablation.csv")
rows = []
for p in ORDER:
    s = a[a.protease == p].set_index("design")
    rows.append(f"{NAME[p]} & " + " & ".join(f"{s.loc[k, 'auroc_vs_background']:.3f} & {s.loc[k, 'auroc_vs_competitors']:.3f}" for k in ["shuffled+background negatives", "background-only negatives", "panel negatives"]) + " \\\\")
write("negative_design", "\\begin{tabular}{@{}lcccccc@{}}\\toprule\n & \\multicolumn{2}{c}{Shuffled + background} & \\multicolumn{2}{c}{Background only} & \\multicolumn{2}{c}{Panel (ours)} \\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\nProtease & vs bkg & vs comp. & vs bkg & vs comp. & vs bkg & vs comp. \\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n")

# ---------------------------------------------------------------- T5 nested + leakage + shuffle + subspace
n = pd.read_csv(R / "methodology" / "nested_regularisation.csv").set_index("protease")
sc = pd.read_csv(R / "methodology" / "vhse_subspace_coverage_summary.csv").set_index("protease")
ls2 = ls.set_index("protease")
rows = []
for p in ORDER:
    rows.append(f"{NAME[p]} & {n.loc[p, 'auroc_fixed_C1']:.3f} & {n.loc[p, 'auroc_nested']:.3f} & {lk.loc[p, 'auroc_random_cv']:.3f} & {lk.loc[p, 'auroc_cluster_cv']:.3f} & {ls2.loc[p, 'auroc_label_shuffled_mean']:.3f} & {100 * sc.loc[p, 'r2_weighted']:.0f}\\% \\\\")
write("controls", "\\begin{tabular}{@{}lcccccc@{}}\\toprule\n & \\multicolumn{2}{c}{Regularisation} & \\multicolumn{2}{c}{Fold construction} & Label & VHSE span \\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\nProtease & $C=1$ & nested & random & clustered & shuffle & coverage \\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n")

# ---------------------------------------------------------------- T6 selectivity matrix
c = pd.read_csv(R / "cv_competitor_auroc.csv")
M = {(r.target, r.competitor): r.auroc for r in c.itertuples()}
rows = []
for t in ORDER:
    rows.append(f"{NAME[t] if t != 'FECAL' else 'Faecal'} & " + " & ".join("--" if t == k else f"{M.get((t, k), np.nan):.2f}" for k in ORDER) + " \\\\")
write("selectivity_matrix", "\\begin{tabular}{@{}l" + "c" * 7 + "@{}}\\toprule\nTarget model & " + " & ".join(["MMP-2", "MMP-8", "MMP-9", "MMP-14", "NE", "LasB", "Faecal"]) + " \\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n")

# ---------------------------------------------------------------- T7 Kridel validation
k = pd.read_csv(R / "validation" / "kridel_scores.csv")
rows = []
for r in k.itertuples():
    km = "not detected" if r.status != "measured" else f"{int(r.kcat_km):,}".replace(",", "{,}")
    rows.append(f"{r.peptide} & \\texttt{{{r.window}}} & {r.n_positions} & {km} & {r.lr_logit:+.2f} & {r.pssm:+.2f} \\\\")
write("kridel_scores", "\\begin{tabular}{@{}llrrrr@{}}\\toprule\nPeptide & Window (P4--P4$'$) & Positions & MMP-9 $k_\\mathrm{cat}/K_m$ (M$^{-1}$s$^{-1}$) & LR logit & PSSM \\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n")
rows = []
for r in kr.itertuples():
    nm = {"lr_logit": "Logistic regression", "pssm": "PSSM", "consensus_z": "Standardised mean"}[r.score]
    rows.append(f"{nm} & {r.set.replace('n=12 incl. C11 as lowest', 'n = 12 (C11 ranked lowest)').replace('n=11 measured', 'n = 11 measured')} & {r.spearman:.2f} & [{r.ci_lo:.2f}, {r.ci_hi:.2f}] & {r.perm_p:.3f} \\\\")
write("kridel_spearman", "\\begin{tabular}{@{}llccc@{}}\\toprule\nScore & Set & Spearman $\\rho$ & 95\\% bootstrap CI & Permutation $p$ \\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n")
mp2 = pd.read_csv(R / "validation" / "kridel_mutant_pairs.csv")
rows = [f"{r.parent}$\\to${r.mutant} & {r.change} & {r.measured_fold:.2f}$\\times$ ({r.measured_log2_fold:+.2f}) & {r.model_delta_logit_bits:+.2f} & {r.model_delta_pssm_bits:+.2f} \\\\" for r in mp2.itertuples()]
write("kridel_mutants", "\\begin{tabular}{@{}llccc@{}}\\toprule\nPair & Substitution & Measured fold change (log$_2$) & LR $\\Delta$ (bits) & PSSM $\\Delta$ (bits) \\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n")

# ---------------------------------------------------------------- T8 structures
st = json.load(open(ROOT / "data" / "structures" / "structures.json"))
rows = []
for p in ["MMP2", "MMP8", "MMP9", "NE", "LASB"]:
    s = st[p]
    rows.append(f"{NAME[p]} & {s['pdb']} & {s['resolution']} & {s['n_residues']} & {', '.join(s['catalytic_metal_ligands'])} & {len(s['pocket_residues'])} \\\\")
write("structures", "\\begin{tabular}{@{}llcclc@{}}\\toprule\nProtease & PDB & Resolution (\\AA) & Residues & Catalytic site residues & Pocket residues \\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n")

# ---------------------------------------------------------------- T9 candidates
top = pd.read_csv(R / "top_candidates_wound_panel.csv"); ref = pd.read_csv(R / "reference_peptides_wound_panel.csv")
mem = ["MMP2", "MMP8", "MMP9", "NE", "LASB"]
def rowf(r):
    return f"\\texttt{{{r.sequence}}} & {NAME[r.target] if r.target != 'FECAL' else 'Faecal'} & " + " & ".join(f"{getattr(r, 'u_' + m):.2f}" for m in mem) + f" & {r.margin:+.2f} \\\\"
hdr = "\\begin{tabular}{@{}llccccc c@{}}\\toprule\nWindow & Target & MMP-2 & MMP-8 & MMP-9 & NE & LasB & Margin \\\\\\midrule\n"
write("candidates", hdr + "\n".join(rowf(r) for r in top.itertuples()) + "\n\\midrule\n" + "\n".join(rowf(r) for r in ref.itertuples()) + "\n\\bottomrule\\end{tabular}\n")
sd = pd.read_csv(ROOT.parent / "substrate_design" / "data" / "outputs" / "candidate_scores.csv")
rows = [f"{r.name.replace('_', ' ')} & \\texttt{{{r.sequence}}} & \\texttt{{{r.window}}} & {r.target} & {r.margin:+.2f} & {r.margin_percentile_in_pssm_pool:.0f} \\\\" for r in sd.itertuples()]
write("design_candidates", "\\begin{tabular}{@{}llllcc@{}}\\toprule\nCandidate & Core & Scored window & Target & Margin (3-plex) & Pool percentile \\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n")

# ---------------------------------------------------------------- T10 data sources
srcs = []
for p in ORDER:
    df = pd.read_csv(ROOT / "data" / "positives" / f"{p}.csv")
    vc = df.source.value_counts()
    srcs.append(f"{NAME[p]} & {vc.get('MEROPS', 0):,} & {vc.get('Eckhard2016', 0):,} & {vc.get('Yang2015', 0):,} & {len(df):,} \\\\".replace(",", "{,}"))
write("data_sources", "\\begin{tabular}{@{}lrrrr@{}}\\toprule\nProtease & MEROPS & Eckhard 2016 & Yang 2015 & Total \\\\\\midrule\n" + "\n".join(srcs) + "\n\\bottomrule\\end{tabular}\n")

# ---------------------------------------------------------------- T11 top residues per position for MMP-9 (illustrative) and learned PSSM peaks
sys_path = str(ROOT); import sys; sys.path.insert(0, sys_path)
from spred.model import load_all
from spred.tables import AMINO_ACIDS
mods = load_all()
rows = []
for p in ORDER:
    lo = mods[p].log_odds
    cells = []
    for i in range(8):
        o = np.argsort(lo[i])[::-1][:3]
        cells.append("".join(AMINO_ACIDS[j] for j in o))
    rows.append(f"{NAME[p] if p != 'FECAL' else 'Faecal'} & " + " & ".join(f"\\texttt{{{c}}}" for c in cells) + " \\\\")
write("top_residues", "\\begin{tabular}{@{}l" + "c" * 8 + "@{}}\\toprule\nProtease & P4 & P3 & P2 & P1 & P1$'$ & P2$'$ & P3$'$ & P4$'$ \\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n")
print("tables:", sorted(x.name for x in T.glob("*.tex")))
