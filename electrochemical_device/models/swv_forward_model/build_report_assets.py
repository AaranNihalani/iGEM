"""Build every figure and number used by reports/tex/swv_forward_model.tex.

Usage:  python build_report_assets.py
Writes reports/figures/swvr_*.pdf and splices the generated number macros and
the sensitivity table into the marked blocks of tex/swv_forward_model.tex
(the only .tex file). Every quantity quoted in the prose is a macro, so the
text cannot drift from the model.
"""
import report_lib as L
import report_figs_device as dev
import report_figs_signal as sig
import report_figs_readout as rd


TEX = L.FIG_DIR.parent / "tex" / "swv_forward_model.tex"


def main():
    dev.build()
    sig.build()
    rd.build()
    L.write_report_blocks(TEX)
    print(len(L.NUMBERS), "numbers written")


if __name__ == "__main__":
    main()
