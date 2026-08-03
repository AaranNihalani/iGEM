"""
mmp9_substrate_selection.py
===========================
Selects the optimal MMP-9 substrate from fluorescence-assay kinetic data.

Clinical target: 17-200 ng/mL MMP-9 in wound fluid (MW = 92 kDa)
                 → 0.185 - 2.17 nM
Incubation time: 30 or 60 min
Model: F_k = 1 - exp(-(kcat_Km_surf) * [E] * T)
       kcat_Km_surf = kcat_Km_sol * alpha  (alpha = surface correction factor)

Selection criterion:
  - F_k(0.185 nM) > 0.05   (detectable above LOD)
  - F_k(2.17 nM)  < 0.90   (not saturated)
  - ΔF_k = F_k(2.17) - F_k(0.185) > 0.15  (calibratable dynamic range)
  - Selectivity: cross-reactivity kcat/Km < 10% of MMP-9 kcat/Km (or absent)

Data from experimental fluorescence assay tables (screenshots).
Cross-reactivity columns are assumed to be competing MMPs (likely MMP-1 / MMP-2).
"""

import numpy as np
import pandas as pd

# ── Substrate data ──────────────────────────────────────────────────────────
# kcat_Km: MMP-9 solution kcat/Km (M⁻¹s⁻¹)
# kcat:    s⁻¹    (None if not reported)
# Km:      mM     (None if not reported)
# cross1, cross2: competing enzyme kcat/Km (M⁻¹s⁻¹); 0 = not detected (**)

substrates = [
    # ID       sequence            kcat_Km   kcat    Km     cross1  cross2
    ("C15",  "SGKGPRQ↓ITA",       188000,   703,    3.7,   15400,  13800),
    ("A11",  "SGKIPRT↓LTA",        67500,  86.4,    1.3,   23500,   7000),
    ("A6",   "SGPRA↓VSTTA",        61000,   178,    2.9,   12400,   1300),
    ("A11m1","SGKIPRR↓LTA",       160000,   224,    1.4,   15200,  24000),
    ("A11m2","SGKIPRT↓ATA",         2000,   9.4,    4.7,       0,      0),
    ("A10",  "SGPLF↓YSVTA",        25600,    20,    0.5,       0,      0),
    ("B1",   "SGQPHY↓LTTA",         8800,     9,    1.0,       0,      0),
    ("A18",  "SGLRPAK↓STA",         5800,    78,   13.5,       0,      0),
    ("A7",   "SG↓LKALMITA",        12400,    30,    2.4,   12300,  41000),
    ("A19",  "SGFGSRY↓LTA",        13200,    11,    0.8,       0,   1000),
    ("A16",  "SGRR↓LLSRTA",         1200,  None,   None,       0,      0),
]

cols = ["ID", "sequence", "kcat_Km_sol", "kcat", "Km_mM", "cross1", "cross2"]
df = pd.DataFrame(substrates, columns=cols)

# ── Constants ───────────────────────────────────────────────────────────────
MW_MMP9  = 92_000          # g/mol
C_LO_ngml = 17             # ng/mL (lower clinical threshold)
C_HI_ngml = 200            # ng/mL (upper clinical threshold)

C_LO_M = (C_LO_ngml * 1e-9 * 1e3) / MW_MMP9   # ng/mL × 1e-9 g/ng × 1e3 mL/L ÷ g/mol = mol/L = 0.185 nM
C_HI_M = (C_HI_ngml * 1e-9 * 1e3) / MW_MMP9   # mol/L = 2.17 nM

T_30  = 30  * 60    # 1800 s
T_60  = 60  * 60    # 3600 s

ALPHAS = [1.0, 0.5, 0.3, 0.1]   # surface correction factors relative to solution

print(f"Clinical range: {C_LO_ngml} ng/mL = {C_LO_M*1e9:.3f} nM  |  "
      f"{C_HI_ngml} ng/mL = {C_HI_M*1e9:.2f} nM\n")

# ── Selectivity score ───────────────────────────────────────────────────────
def selectivity_ok(row, threshold=0.10):
    """True if both cross-reactivities are < threshold × MMP-9 kcat/Km."""
    kk = row["kcat_Km_sol"]
    for c in [row["cross1"], row["cross2"]]:
        if c > threshold * kk:
            return False
    return True

df["selective"] = df.apply(selectivity_ok, axis=1)

# ── F_k calculation ─────────────────────────────────────────────────────────
def fk(kcat_Km_surf, c_M, T_s):
    k_obs = kcat_Km_surf * c_M
    return 1 - np.exp(-k_obs * T_s)

results = []
for _, row in df.iterrows():
    kk_sol = row["kcat_Km_sol"]
    for alpha in ALPHAS:
        kk_surf = kk_sol * alpha
        for T_label, T_s in [("30 min", T_30), ("60 min", T_60)]:
            flo = fk(kk_surf, C_LO_M, T_s)
            fhi = fk(kk_surf, C_HI_M, T_s)
            delta = fhi - flo
            viable = (flo > 0.05) and (fhi < 0.90) and (delta > 0.15)
            results.append({
                "ID":          row["ID"],
                "sequence":    row["sequence"],
                "kcat_Km_sol": kk_sol,
                "alpha":       alpha,
                "kcat_Km_surf":kk_surf,
                "T":           T_label,
                "F_lo (0.185nM)": round(flo, 4),
                "F_hi (2.17nM)":  round(fhi, 4),
                "ΔF_k":           round(delta, 4),
                "viable":      viable,
                "selective":   row["selective"],
            })

res = pd.DataFrame(results)

# ── Print full table ────────────────────────────────────────────────────────
print("="*110)
print(f"{'ID':<8} {'α':>5} {'T':>8} {'kcat/Km_surf':>14} "
      f"{'F(0.18nM)':>11} {'F(2.17nM)':>10} {'ΔF_k':>7} "
      f"{'viable':>7} {'selective':>10}")
print("-"*110)

for _, r in res.iterrows():
    print(f"{r['ID']:<8} {r['alpha']:>5.1f} {r['T']:>8} {r['kcat_Km_surf']:>14,.0f} "
          f"{r['F_lo (0.185nM)']:>10.1%} {r['F_hi (2.17nM)']:>10.1%} "
          f"{r['ΔF_k']:>7.1%} "
          f"{'YES' if r['viable'] else 'no':>7} "
          f"{'YES' if r['selective'] else 'no':>10}")
    if r["T"] == "60 min" and r["alpha"] == ALPHAS[-1]:
        print()

# ── Viable + selective candidates ───────────────────────────────────────────
print("="*110)
print("\nCANDIDATES THAT ARE BOTH VIABLE AND SELECTIVE:")
print("="*110)
good = res[res["viable"] & res["selective"]].copy()
if good.empty:
    print("  None under strict criteria. Relaxing ΔF_k threshold to 0.08...")
    good = res[(res["F_lo (0.185nM)"] > 0.03) &
               (res["F_hi (2.17nM)"] < 0.90) &
               (res["ΔF_k"] > 0.08) &
               res["selective"]].copy()

good_sorted = good.sort_values(["ΔF_k"], ascending=False)
print(f"{'ID':<8} {'seq':<22} {'α':>5} {'T':>8} {'F_lo':>8} {'F_hi':>8} {'ΔF_k':>7}")
print("-"*75)
for _, r in good_sorted.head(15).iterrows():
    print(f"{r['ID']:<8} {r['sequence']:<22} {r['alpha']:>5.1f} {r['T']:>8} "
          f"{r['F_lo (0.185nM)']:>7.1%} {r['F_hi (2.17nM)']:>8.1%} "
          f"{r['ΔF_k']:>7.1%}")

# ── Comparison: where does our current choice A11 (IPRTLTAA) sit? ────────────
print("\n" + "="*110)
print("OUR CURRENT SUBSTRATE: A11 (SGKIPRT↓LTA → IPRTLTAA core)")
print("="*110)
a11 = res[res["ID"] == "A11"]
print(f"  selective = {a11.iloc[0]['selective']}  "
      f"(cross1={df[df.ID=='A11'].cross1.values[0]:,}  "
      f"cross2={df[df.ID=='A11'].cross2.values[0]:,}  "
      f"MMP-9 kcat/Km={df[df.ID=='A11'].kcat_Km_sol.values[0]:,})")
print()
print(f"  {'α':>5} {'T':>8} {'F_lo':>9} {'F_hi':>9} {'ΔF_k':>7} {'viable':>8}")
for _, r in a11.iterrows():
    print(f"  {r['alpha']:>5.1f} {r['T']:>8} "
          f"{r['F_lo (0.185nM)']:>9.1%} {r['F_hi (2.17nM)']:>9.1%} "
          f"{r['ΔF_k']:>7.1%} {'YES' if r['viable'] else 'no':>8}")

# ── Summary recommendation ───────────────────────────────────────────────────
print("\n" + "="*110)
print("RECOMMENDATION SUMMARY")
print("="*110)
print(f"  Clinical range:  {C_LO_M*1e9:.2f} – {C_HI_M*1e9:.2f} nM  ({C_LO_ngml}–{C_HI_ngml} ng/mL, MMP-9 MW={MW_MMP9:,} Da)")
print(f"  LOD target:      F_k > 5% at lower bound")
print(f"  Dynamic range:   F_k < 90% at upper bound")
print(f"  ΔF_k target:     > 15% across full range (relaxed: > 8%)")
print()
if not good_sorted.empty:
    best = good_sorted.iloc[0]
    print(f"  ★ Best candidate: {best['ID']}  |  sequence: {best['sequence']}")
    print(f"    Surface correction α = {best['alpha']}  |  incubation = {best['T']}")
    print(f"    F_k range: {best['F_lo (0.185nM)']:.1%} → {best['F_hi (2.17nM)']:.1%}  "
          f"(ΔF_k = {best['ΔF_k']:.1%})")
