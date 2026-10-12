"""
export_peak_data.py -- writes the MB/Fc/BSA peak data behind
max_peaks_plots.py to an Excel workbook (one sheet per condition) and
matching CSVs, for Max.

Must be run after / alongside max_peaks_plots.py (imports its extraction
functions directly, so the numbers in the plots and in this export can
never drift apart).
"""
from pathlib import Path

import openpyxl
from openpyxl.styles import Font

from max_peaks_plots import gather, gather_bsa, normalise

OUT = Path(__file__).parent / "max_peaks_figures"
OUT.mkdir(exist_ok=True)

HEADER_MB_FC = ["File #", "Time (min)", "MB peak (nA)", "MB peak potential (mV)", "MB included",
                "MB % of first scan", "Fc peak (nA)", "Fc peak potential (mV)", "Fc included",
                "Fc % of first scan", "MB/Fc ratio", "Ratio included", "Notes"]
HEADER_BSA = ["File #", "Time (min)", "Signal peak (nA)", "Signal peak potential (mV)",
              "Signal % of first scan", "Notes"]


def condition_table(rows, label):
    out = []
    for r in rows:
        if r["missing"]:
            out.append([r["i"], r["t"], None, None, None, None, None, None, None, None, None, None,
                        "missing: empty .xlsx export"])
            continue
        ratio_ok = r["mb_ok"] and r["fc_ok"]
        ratio = (r["mb"] / r["fc"]) if (ratio_ok and r["mb"] is not None and r["fc"] is not None) else None
        notes = []
        if not r["mb_ok"]:
            notes.append("MB excluded: sweep starts at -200 mV, clips the MB peak")
        if not r["fc_ok"]:
            notes.append("Fc excluded: sweep starts at -200 mV, clips the Fc peak")
        out.append([
            r["i"], r["t"],
            round(r["mb"], 3), round(r["mb_pot"], 1), "Y" if r["mb_ok"] else "N", round(r["mb_pct"], 1) if r["mb_pct"] is not None else None,
            round(r["fc"], 3), round(r["fc_pot"], 1), "Y" if r["fc_ok"] else "N", round(r["fc_pct"], 1) if r["fc_pct"] is not None else None,
            round(ratio, 3) if ratio is not None else None, "Y" if ratio_ok else "N",
            "; ".join(notes) if notes else "ok",
        ])
    return out


def bsa_table(rows):
    out = []
    for r in rows:
        if r["missing"]:
            out.append([r["i"], r["t"], None, None, None, "missing: empty .xlsx export"])
            continue
        out.append([r["i"], r["t"], round(r["sig"], 3), round(r["sig_pot"], 1),
                    round(r["sig_pct"], 1) if r["sig_pct"] is not None else None,
                    "not MB or Fc (outside both their windows); real local peak, -50 to +150 mV window"])
    return out


def write_sheet(wb, title, header, rows, note):
    ws = wb.create_sheet(title)
    ws["A1"] = title
    ws["A1"].font = Font(bold=True, size=13)
    ws["A2"] = note
    ws.append([])
    ws.append(header)
    for cell in ws[4]:
        cell.font = Font(bold=True)
    for row in rows:
        ws.append(row)
    for col in ws.columns:
        width = max((len(str(c.value)) for c in col if c.value is not None), default=10)
        ws.column_dimensions[col[0].column_letter].width = min(width + 2, 60)


if __name__ == "__main__":
    cleavable = gather("MMP-9 20 nM", mb_valid_scan1=True, fc_valid_scan1=True)
    normalise(cleavable, "mb"); normalise(cleavable, "fc")

    noncleavable = gather("MMP-9 20nM", mb_valid_scan1=False, fc_valid_scan1=True)
    normalise(noncleavable, "mb"); normalise(noncleavable, "fc")

    bsa = gather_bsa()

    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    write_sheet(wb, "Cleavable (MMP-9 20 nM)", HEADER_MB_FC, condition_table(cleavable, "cleavable"),
                "Peak heights from net SWV current (nA). Time = (file index - 1) x 5 min. "
                "MB window -260 to -100 mV, Fc window +150 to +400 mV, peak = maximum in window. "
                "% columns are relative to that channel's own first included scan (= 100%). "
                "Scan 1 is valid for both MB and Fc here -- not clipped.")
    write_sheet(wb, "Non-cleavable (MMP-9 20nM)", HEADER_MB_FC, condition_table(noncleavable, "noncleavable"),
                "Peak heights from net SWV current (nA). Time = (file index - 1) x 5 min. "
                "MB window -260 to -100 mV, Fc window +150 to +400 mV, peak = maximum in window. "
                "% columns are relative to that channel's own first included scan (= 100%). "
                "Scan 1 MB is clipped (sweep starts at -200 mV) and excluded; scan 1 Fc is valid.")
    write_sheet(wb, "BSA control", HEADER_BSA, bsa_table(bsa),
                "BSA control has no MB or Fc peak -- its one real, reproducible feature sits near "
                "0-100 mV, clear of both the MB and Fc windows above. Tracked here as a generic "
                "signal peak in a -50 to +150 mV window, % of its own first scan.")

    xlsx_path = OUT / "max_peaks_data.xlsx"
    wb.save(xlsx_path)
    print("saved", xlsx_path)

    import csv
    for title, header, rows in [
        ("cleavable_peaks.csv", HEADER_MB_FC, condition_table(cleavable, "cleavable")),
        ("noncleavable_peaks.csv", HEADER_MB_FC, condition_table(noncleavable, "noncleavable")),
        ("bsa_control_peaks.csv", HEADER_BSA, bsa_table(bsa)),
    ]:
        path = OUT / title
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
        print("saved", path)
