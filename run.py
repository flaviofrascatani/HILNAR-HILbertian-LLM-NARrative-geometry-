"""
HILNAR prototype — 2019 & 2025 Belarusian corpus.
Computes NSI (diachronic, 2019→2025) and SAI/TDI (synchronic) for all four
carriers, plus permutation-based significance tests for both indices.

Usage:
    python run.py

Outputs (written to /mnt/user-data/outputs/):
    hilnar_full_results.json   — all numeric results
    hilnar_full_indices.png    — bar chart (NSI) + line chart (TDI)
"""

import json
import math
import os
import random
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import hilnar as H
from hilnar import CARRIER_NAMES
from corpus_full import DOCUMENTS

OUT = "/mnt/user-data/outputs"
os.makedirs(OUT, exist_ok=True)

CARRIERS = ["g", "m", "e", "p"]
NON_GOV   = ["m", "e", "p"]
DIM       = 20
MINCOUNT  = 2
STANDARD  = ("g", 2019)
N_PERM    = 300


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline helpers
# ──────────────────────────────────────────────────────────────────────────────
def build(docs=None, dim=DIM, mincount=MINCOUNT, standard=STANDARD):
    if docs is None:
        docs = DOCUMENTS
    docs, dedup = H.deduplicate(docs)
    tok = H.train_tokenizer(docs)
    docs = H.tokenize_docs(docs, tok)
    counter = Counter()
    for d in docs:
        counter.update(d["_tokens"])
    vocab = [w for w, c in counter.items()
             if c >= mincount and w not in ("[UNK]", "[PAD]")]
    slices  = H.backend_cooc(docs, vocab, dim=dim)
    reduced = H.project_and_align(
        slices, vocab, standard, dim=dim,
        basis_source="pooled", procrustes=False,
    )
    return docs, dedup, vocab, slices, reduced


def nsi_carrier(reduced, slices, carrier, t=2019, tp=2025):
    kt, ktp = (carrier, t), (carrier, tp)
    if kt not in reduced or ktp not in reduced:
        return float("nan"), 0
    wt, wtp  = reduced[kt], reduced[ktp]
    ft       = slices[kt].freqs
    shared   = [w for w in wt if w in wtp]
    num = den = 0.0
    for w in shared:
        f    = H.logf(ft.get(w, 0))
        num += f * H.angular_displacement(wt[w], wtp[w])
        den += f
    return (num / den if den else float("nan")), len(shared)


def sai_val(reduced, slices, sector, year):
    ks, kg = (sector, year), ("g", year)
    if ks not in reduced or kg not in reduced:
        return float("nan"), 0
    ws, wg = reduced[ks], reduced[kg]
    fs     = slices[ks].freqs
    shared = [w for w in ws if w in wg]
    num = den = 0.0
    for w in shared:
        f    = H.logf(fs.get(w, 0))
        num += f * H._cos(ws[w], wg[w])
        den += f
    return (num / den if den else float("nan")), len(shared)


# ──────────────────────────────────────────────────────────────────────────────
# Permutation tests
# ──────────────────────────────────────────────────────────────────────────────
def perm_nsi(n_perm=N_PERM, seed=0):
    """Null for NSI: shuffle year labels within each carrier."""
    rng = random.Random(seed)
    _, _, _, slices0, reduced0 = build()
    obs = {c: nsi_carrier(reduced0, slices0, c)[0] for c in CARRIERS}
    null = {c: [] for c in CARRIERS}
    for _ in range(n_perm):
        perm = []
        for c in CARRIERS:
            ds  = [dict(d) for d in DOCUMENTS if d["carrier"] == c]
            yrs = [d["year"] for d in ds]
            rng.shuffle(yrs)
            for d, y in zip(ds, yrs):
                d["year"] = y
            perm += ds
        _, _, _, sl, red = build(docs=perm)
        for c in CARRIERS:
            v, _ = nsi_carrier(red, sl, c)
            if not math.isnan(v):
                null[c].append(v)
    out = {}
    for c in CARRIERS:
        arr = np.array(null[c]) if null[c] else np.array([float("nan")])
        p   = float(np.mean(arr >= obs[c])) if not math.isnan(obs[c]) else float("nan")
        out[c] = {
            "observed":  float(obs[c]),
            "null_mean": float(np.nanmean(arr)),
            "null_sd":   float(np.nanstd(arr)),
            "p_value":   p,
        }
    return out


def perm_sai(year, n_perm=N_PERM, seed=1):
    """Null for SAI: hold government fixed, shuffle non-gov carrier labels."""
    rng  = random.Random(seed)
    _, _, _, sl0, red0 = build()
    obs  = {s: sai_val(red0, sl0, s, year)[0] for s in NON_GOV}
    gov  = [d for d in DOCUMENTS if d["carrier"] == "g"]
    nong = [d for d in DOCUMENTS if d["carrier"] != "g"]
    sizes = {s: sum(1 for d in nong if d["carrier"] == s) for s in NON_GOV}
    null  = {s: [] for s in NON_GOV}
    for _ in range(n_perm):
        sh = [dict(d) for d in nong]
        rng.shuffle(sh)
        i = 0
        for s in NON_GOV:
            for _ in range(sizes[s]):
                sh[i]["carrier"] = s
                i += 1
        _, _, _, sl, red = build(docs=gov + sh)
        for s in NON_GOV:
            v, _ = sai_val(red, sl, s, year)
            if not math.isnan(v):
                null[s].append(v)
    out = {}
    for s in NON_GOV:
        arr = np.array(null[s]) if null[s] else np.array([float("nan")])
        p   = float(np.mean(np.abs(arr) >= abs(obs[s]))) if not math.isnan(obs[s]) else float("nan")
        out[s] = {
            "observed":           float(obs[s]),
            "null_mean":          float(np.nanmean(arr)),
            "null_sd":            float(np.nanstd(arr)),
            "p_value_two_sided":  p,
        }
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    docs, dedup, vocab, slices, reduced = build()
    ndocs = {
        f"{c}-{y}": slices[(c, y)].n_docs
        for c in CARRIERS for y in (2019, 2025)
        if (c, y) in slices
    }

    sep = "=" * 74
    print(sep)
    print("HILNAR — 2019 + 2025 Belarusian corpus")
    print(sep)
    print(f"Docs per slice   : {ndocs}")
    print(f"Analysis vocab   : {len(vocab)} tokens (freq ≥ {MINCOUNT})")

    # ── (A) NSI ───────────────────────────────────────────────────────────────
    print("-" * 74)
    print("(A) NSI — diachronic shift 2019→2025  [0 = none … 1 = inversion]")
    nsi_vals = {}
    for c in CARRIERS:
        v, sh = nsi_carrier(reduced, slices, c)
        nsi_vals[c] = v
        miss = "  (one year missing)" if not ((c,2019) in slices and (c,2025) in slices) else ""
        if not math.isnan(v):
            print(f"  {CARRIER_NAMES[c]:<18}: NSI={v:.3f}  (shared={sh}){miss}")
        else:
            print(f"  {CARRIER_NAMES[c]:<18}: n/a{miss}")

    num = den = 0.0
    for c in CARRIERS:
        if not math.isnan(nsi_vals.get(c, float("nan"))):
            w    = H.logf(slices[(c, 2019)].n_docs)
            num += w * nsi_vals[c]
            den += w
    state_nsi = num / den if den else float("nan")
    print(f"  {'STATE (weighted)':<18}: NSI={state_nsi:.3f}")

    # ── (B) SAI / TDI ─────────────────────────────────────────────────────────
    print("-" * 74)
    print("(B) SAI vs government  [-1 anti-align … +1 mirror]  → 2-point TDI")
    sai_tab = {}
    for y in (2019, 2025):
        row = []
        for s in NON_GOV:
            v, sh = sai_val(reduced, slices, s, y)
            sai_tab[(s, y)] = v
            row.append(f"{CARRIER_NAMES[s][:5]}={v:+.3f}(n={sh})")
        comp = float(np.nanmean([sai_tab[(s, y)] for s in NON_GOV]))
        print(f"  {y}:  " + "  ".join(row) + f"  | TDI_composite={comp:+.3f}")

    # ── (C) NSI significance ──────────────────────────────────────────────────
    print("-" * 74)
    print(f"(C) NSI permutation test  (n={N_PERM}, shuffle year labels):")
    pn = perm_nsi()
    for c in CARRIERS:
        d = pn[c]
        if not math.isnan(d["observed"]):
            print(f"  {CARRIER_NAMES[c]:<18}: NSI={d['observed']:.3f}  "
                  f"null={d['null_mean']:.3f}±{d['null_sd']:.3f}  p={d['p_value']:.2f}")

    # ── (D) SAI significance ──────────────────────────────────────────────────
    print("-" * 74)
    print(f"(D) SAI permutation test  (n={N_PERM}, shuffle carrier labels):")
    sai_perm = {}
    for y in (2019, 2025):
        ps = perm_sai(y)
        sai_perm[y] = ps
        print(f"  --- {y} ---")
        for s in NON_GOV:
            d = ps[s]
            print(f"    {CARRIER_NAMES[s]:<18}: SAI={d['observed']:+.3f}  "
                  f"null={d['null_mean']:+.3f}±{d['null_sd']:.3f}  "
                  f"p={d['p_value_two_sided']:.2f}")

    # ── Charts ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    cs = [c for c in CARRIERS if not math.isnan(nsi_vals.get(c, float("nan")))]
    bar_labels  = [CARRIER_NAMES[c] for c in cs] + ["STATE"]
    bar_vals    = [nsi_vals[c]      for c in cs] + [state_nsi]
    bar_colors  = ["#4C72B0"] * len(cs) + ["#C44E52"]
    axes[0].bar(bar_labels, bar_vals, color=bar_colors)
    axes[0].set_title("(A) NSI: narrative shift 2019→2025")
    axes[0].set_ylabel("NSI [0, 1]")
    for i, v in enumerate(bar_vals):
        axes[0].text(i, v + 0.005, f"{v:.2f}", ha="center", fontsize=9)
    axes[0].tick_params(axis="x", rotation=15)

    palette = {"m": "#4C72B0", "e": "#55A868", "p": "#C44E52"}
    for s in NON_GOV:
        axes[1].plot(
            [2019, 2025],
            [sai_tab[(s, 2019)], sai_tab[(s, 2025)]],
            marker="o", color=palette[s], label=CARRIER_NAMES[s],
        )
    axes[1].axhline(0, color="#888", lw=0.8, ls="--")
    axes[1].set_title("(B) SAI vs government: TDI trajectory")
    axes[1].set_ylabel("SAI [−1, +1]")
    axes[1].set_xticks([2019, 2025])
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT}/hilnar_full_indices.png", dpi=150)
    plt.close()

    # ── JSON export ───────────────────────────────────────────────────────────
    result = {
        "docs_per_slice": ndocs,
        "vocab_size":     len(vocab),
        "NSI": {c: float(nsi_vals.get(c, float("nan"))) for c in CARRIERS},
        "state_NSI":      float(state_nsi),
        "SAI": {
            f"{s}-{y}": float(sai_tab.get((s, y), float("nan")))
            for s in NON_GOV for y in (2019, 2025)
        },
        "TDI_2019": float(np.nanmean([sai_tab.get((s, 2019), float("nan")) for s in NON_GOV])),
        "TDI_2025": float(np.nanmean([sai_tab.get((s, 2025), float("nan")) for s in NON_GOV])),
        "NSI_permutation": pn,
        "SAI_permutation": {str(y): sai_perm[y] for y in (2019, 2025)},
        "sources": [
            {"carrier": d["carrier"], "year": d["year"],
             "source": d["source"],   "note": d["note"]}
            for d in DOCUMENTS
        ],
    }
    path = f"{OUT}/hilnar_full_results.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=float)

    print(sep)
    print("Saved hilnar_full_results.json  +  hilnar_full_indices.png")
    print(sep)
    return result


if __name__ == "__main__":
    main()
