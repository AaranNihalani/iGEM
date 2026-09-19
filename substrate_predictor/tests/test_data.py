import numpy as np
import pandas as pd
import pytest

from spred import data
from spred.constants import PROTEASES, WINDOW


def test_positives_are_valid_windows_and_unique():
    for pid in PROTEASES:
        seqs = pd.read_csv(data.POSITIVES_DIR / f"{pid}.csv")["sequence"]
        assert seqs.is_unique and all(data.is_window(s) for s in seqs)


def test_yang_table_consistent_with_uniprot_sequence():
    assert len(data.yang_positives()) > 50      # raises if any row is inconsistent with P04985


def test_negatives_exclude_target_positives_and_are_deterministic():
    pos = data.trainable_positives()
    a = data.make_negatives("MMP9", pos)
    b = data.make_negatives("MMP9", pos)
    assert a.equals(b)
    assert not set(a.sequence) & set(pos["MMP9"])
    assert len(a) == 3 * len(pos["MMP9"]) or len(a) > 0.95 * 3 * len(pos["MMP9"])
    assert a.sequence.is_unique


def test_panel_negatives_contain_each_competitor():
    pos = data.trainable_positives()
    src = set(data.make_negatives("MMP9", pos).source)
    assert set(pos) - {"MMP9"} <= src and "background" in src


def test_baseline_negatives_are_composition_matched_shuffles():
    pos = data.trainable_positives()
    neg = data.make_negatives("NE", pos, mode="baseline")
    sh = neg[neg.source == "shuffled"].sequence
    comps = {"".join(sorted(s)) for s in pos["NE"]}
    assert len(sh) > 0 and all("".join(sorted(s)) in comps for s in sh)


@pytest.mark.skipif(not any(p.exists() for p in [data.DATA / "raw" / "Substrate_search.txt",
                                                  data.ROOT.parent / "cross_reactivity_matrix" / "data" / "raw" / "Substrate_search.txt"]),
                    reason="MEROPS file not available")
def test_positives_reproduce_from_sources():
    rebuilt = data.build_positives()
    for pid, df in rebuilt.items():
        saved = pd.read_csv(data.POSITIVES_DIR / f"{pid}.csv")
        assert set(df.sequence) == set(saved.sequence), pid
