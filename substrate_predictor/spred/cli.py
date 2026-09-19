"""Command line interface:  python -m spred <command> ..."""
import argparse
import sys

import pandas as pd


def main(argv=None):
    ap = argparse.ArgumentParser(prog="spred", description="MMP/protease substrate predictor")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build-data", help="rebuild data/positives/ from source files")
    t = sub.add_parser("train", help="cross-validate and train all models")
    t.add_argument("--boot", type=int, default=500)
    p = sub.add_parser("predict", help="score one or more 8-mers against a panel")
    p.add_argument("peptides", nargs="+")
    p.add_argument("--panel", default="wound")
    r = sub.add_parser("rank", help="rank candidate peptides for a target protease")
    r.add_argument("--target", required=True)
    r.add_argument("--panel", default="wound")
    r.add_argument("--pool", type=int, default=50000, help="random pool size")
    r.add_argument("--top", type=int, default=15)
    m = sub.add_parser("mutate", help="score every single substitution of a peptide")
    m.add_argument("peptide")
    m.add_argument("--target", required=True)
    m.add_argument("--panel", default="wound")
    m.add_argument("--top", type=int, default=15)
    s = sub.add_parser("scan", help="slide an 8-mer window over a FASTA file")
    s.add_argument("fasta")
    s.add_argument("--target", required=True)
    s.add_argument("--panel", default="wound")
    s.add_argument("--top", type=int, default=15)
    a = ap.parse_args(argv)

    pd.set_option("display.width", 200, "display.max_columns", 30)
    if a.cmd == "build-data":
        from . import data
        pos = data.build_positives()
        data.write_positives(pos)
        for k, v in pos.items():
            print(f"{k:6s} {len(v):5d}  {v.source.value_counts().to_dict()}")
        return
    if a.cmd == "train":
        from .train import train_all
        train_all(n_boot=a.boot)
        return

    from . import candidates
    from .panel import PRESETS, Panel
    panel = Panel(PRESETS.get(a.panel, a.panel.split(",")))
    if a.cmd == "predict":
        print(panel.score([x.upper() for x in a.peptides]).round(3).to_string(index=False))
    elif a.cmd == "rank":
        seqs = candidates.random_pool(a.pool)
        print(panel.rank(seqs, a.target).head(a.top).round(3).to_string(index=False))
    elif a.cmd == "mutate":
        muts = candidates.single_mutants(a.peptide)
        sc = panel.score([m[2] for m in muts])
        sc["margin"] = panel.margin(sc, a.target)
        sc.insert(1, "change", [f"{a.peptide[i]}{i + 1}{aa}" for i, aa, _ in muts])
        print(sc.sort_values("margin", ascending=False).head(a.top).round(3).to_string(index=False))
    elif a.cmd == "scan":
        w = candidates.fasta_windows(a.fasta)
        sc = panel.score([x[2] for x in w])
        sc["margin"] = panel.margin(sc, a.target)
        sc.insert(0, "where", [f"{n}:{p}" for n, p, _ in w])
        print(sc.sort_values("margin", ascending=False).head(a.top).round(3).to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
