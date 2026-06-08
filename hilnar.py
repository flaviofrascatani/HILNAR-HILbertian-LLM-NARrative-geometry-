"""
HILNAR — Hilbertian-LLM Narrative Geometry — PROTOTIPE FOR THE THESIS (BETA VERSION) NOT COMPLETE HILNAR, 
NO WAVE PROJECTION (LOOK AT HILNAR_WAVE.PY FOR THE WAVE PROJECTION. THIS LATTER (WAVE) HOWEVER HAS NOT BEEN IMPLEMENTED FOR COMPUTING POWER REQUIRED. ONLY THIS HILNAR.PY PROTOTYPE HAS BEEN IMPLEMENTED). 

Implements the four phases described in the thesis:

  1. Tokenizing phase   : carrier split, exact + quasi dedup (hash + Jaccard
                          shingles), WordPiece subwording, logged frequency.
  2. Transformer phase  : two interchangeable embedding backends
                          - "e5"      : multilingual-E5 contextual mean-pooling
                                        (faithful to the thesis; needs the HF
                                        model, so it is NOT runnable offline)
                          - "cooc"    : PPMI co-occurrence + SVD static
                                        embeddings, trained PER (sector, year)
                                        slice -> genuinely separate spaces ->
                                        the Procrustes machinery is meaningful.
                                        Runs fully offline. DEFAULT.
  3. Hilbert projection : truncated-SVD reduction with a SHARED basis taken
                          from the standard space, then orthogonal Procrustes
                          alignment of every slice to the standard space.
  4. Comparative phase  : NSI (diachronic) and SAI/TDI (synchronic).

Point projection only (the thesis defers the wave / quantum-probability
variant to future work for storage/compute reasons).

Author-side note: anisotropy correction (global mean-centring) is applied
before cosine-based quantities — see ENGINEERING_NOTES in the accompanying
report for why this is necessary and is an addition to the thesis text.
"""

from __future__ import annotations
import hashlib
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable

import numpy as np
from scipy.linalg import orthogonal_procrustes
from sklearn.cluster import AgglomerativeClustering

from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Lowercase, NFC, Sequence as NormSeq

CARRIERS = ["g", "m", "e", "p"]
CARRIER_NAMES = {"g": "government", "m": "media", "e": "education", "p": "popular culture"}


# --------------------------------------------------------------------------- #
# Phase 1 — Tokenizing
# --------------------------------------------------------------------------- #
def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _shingles(text: str, k: int = 3) -> set[str]:
    words = text.split()
    if len(words) < k:
        return {" ".join(words)}
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def deduplicate(docs: list[dict], jaccard_thr: float = 0.9) -> tuple[list[dict], dict]:
    """Exact dedup (sha256 of normalised text) then quasi dedup (Jaccard on
    word shingles), applied within each (carrier, year) stream."""
    report = {"exact_removed": 0, "quasi_removed": 0}
    kept: list[dict] = []
    seen_hash: set[str] = set()
    by_stream: dict[tuple, list[set]] = defaultdict(list)

    for d in docs:
        norm = _normalize(d["text"])
        h = hashlib.sha256(norm.encode("utf-8")).hexdigest()
        if h in seen_hash:
            report["exact_removed"] += 1
            continue
        sh = _shingles(norm)
        stream = (d["carrier"], d["year"])
        if any(_jaccard(sh, prev) >= jaccard_thr for prev in by_stream[stream]):
            report["quasi_removed"] += 1
            continue
        seen_hash.add(h)
        by_stream[stream].append(sh)
        kept.append({**d, "_norm": norm})
    return kept, report


def train_tokenizer(docs: list[dict], vocab_size: int = 3000, min_freq: int = 1) -> Tokenizer:
    """Train ONE WordPiece tokenizer on the whole corpus so the subword
    vocabulary is shared across every slice (required for NSI/SAI overlap)."""
    tok = Tokenizer(WordPiece(unk_token="[UNK]"))
    tok.normalizer = NormSeq([NFC(), Lowercase()])
    tok.pre_tokenizer = Whitespace()
    trainer = WordPieceTrainer(
        vocab_size=vocab_size,
        min_frequency=min_freq,
        special_tokens=["[UNK]", "[PAD]"],
        continuing_subword_prefix="##",
    )
    tok.train_from_iterator([d["_norm"] for d in docs], trainer=trainer)
    return tok


def tokenize_docs(docs: list[dict], tok: Tokenizer) -> list[dict]:
    for d in docs:
        d["_tokens"] = tok.encode(d["_norm"]).tokens
    return docs


# --------------------------------------------------------------------------- #
# Phase 2 — Embedding backends. Each returns, per (carrier, year) slice:
#   vectors : dict[subword -> np.ndarray]   (one vector per subword type)
#   freqs   : dict[subword -> int]          (raw subword count in the slice)
# --------------------------------------------------------------------------- #
@dataclass
class Slice:
    carrier: str
    year: int
    vectors: dict[str, np.ndarray]
    freqs: dict[str, int]
    n_docs: int


def _slice_token_streams(docs: list[dict]) -> dict[tuple, list[list[str]]]:
    streams: dict[tuple, list[list[str]]] = defaultdict(list)
    for d in docs:
        streams[(d["carrier"], d["year"])].append(d["_tokens"])
    return streams


def backend_cooc(docs: list[dict], vocab: list[str], dim: int = 50,
                 window: int = 5) -> dict[tuple, Slice]:
    """PPMI co-occurrence embeddings, one space per (carrier, year) slice.

    Each slice gets its own PPMI(V x V) matrix; the SHARED projection basis is
    taken later (in the projection phase) from the standard slice, so here we
    just return the full PPMI rows as the slice's raw embedding. Dimensionality
    reduction + alignment happen in Phase 3, exactly as the thesis prescribes.
    """
    vidx = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)
    streams = _slice_token_streams(docs)
    slices: dict[tuple, Slice] = {}

    for (carrier, year), token_lists in streams.items():
        co = np.zeros((V, V), dtype=np.float64)
        freqs: Counter = Counter()
        for toks in token_lists:
            idxs = [vidx[t] for t in toks if t in vidx]
            for t in toks:
                if t in vidx:
                    freqs[t] += 1
            for i, ci in enumerate(idxs):
                lo, hi = max(0, i - window), min(len(idxs), i + window + 1)
                for j in range(lo, hi):
                    if j != i:
                        co[ci, idxs[j]] += 1.0
        # symmetric PPMI
        co = (co + co.T) / 2.0
        total = co.sum()
        if total == 0:
            continue
        row = co.sum(axis=1, keepdims=True)
        col = co.sum(axis=0, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            pmi = np.log((co * total) / (row @ col))
        ppmi = np.nan_to_num(np.maximum(pmi, 0.0), nan=0.0, posinf=0.0, neginf=0.0)
        vectors = {w: ppmi[vidx[w]].copy() for w in freqs}  # raw V-dim rows
        slices[(carrier, year)] = Slice(carrier, year, vectors, dict(freqs), len(token_lists))
    return slices


def backend_e5(docs: list[dict], vocab: list[str], **_) -> dict[tuple, Slice]:
    """Faithful-to-thesis contextual backend: multilingual-E5, mean-pool the
    contextual vectors of every occurrence of each subword across the slice.

    NOT runnable in this sandbox (needs the HuggingFace model). Kept so the
    production pipeline only has to flip backend="e5".
    """
    from transformers import AutoTokenizer, AutoModel  # noqa
    import torch  # noqa

    name = "intfloat/multilingual-e5-base"
    hf_tok = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name).eval()
    streams: dict[tuple, list[str]] = defaultdict(list)
    for d in docs:
        streams[(d["carrier"], d["year"])].append(d["_norm"])

    slices: dict[tuple, Slice] = {}
    for (carrier, year), texts in streams.items():
        acc: dict[str, list[np.ndarray]] = defaultdict(list)
        for t in texts:
            enc = hf_tok(t, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                out = model(**enc).last_hidden_state[0]
            ids = enc["input_ids"][0].tolist()
            sub = hf_tok.convert_ids_to_tokens(ids)
            for tokstr, vec in zip(sub, out):
                acc[tokstr.replace("\u2581", "")].append(vec.numpy())
        vectors = {w: np.mean(v, axis=0) for w, v in acc.items() if w}
        freqs = {w: len(v) for w, v in acc.items() if w}
        slices[(carrier, year)] = Slice(carrier, year, vectors, freqs, len(texts))
    return slices


BACKENDS: dict[str, Callable] = {"cooc": backend_cooc, "e5": backend_e5}


# --------------------------------------------------------------------------- #
# Phase 3 — Hilbert projection: shared-basis SVD reduction + Procrustes
# --------------------------------------------------------------------------- #
def _matrix(slice_: Slice, vocab: list[str]) -> tuple[np.ndarray, list[str]]:
    words = [w for w in vocab if w in slice_.vectors]
    M = np.vstack([slice_.vectors[w] for w in words])
    return M, words


def project_and_align(slices: dict[tuple, Slice], vocab: list[str],
                      standard_key: tuple, dim: int = 50,
                      basis_source: str = "pooled", procrustes: bool = False,
                      center: bool = False):
    """Map every slice into one common reduced frame.

    basis_source : "pooled"   -> truncated-SVD basis from ALL slices stacked
                                 (preserves period/sector-specific shared
                                 directions; this is the CORRECTED default).
                   "standard"  -> basis from the standard slice only
                                 (the original thesis text; kept for
                                 reproducibility — it inverted SAI on the test
                                 corpus).
    procrustes   : per-slice orthogonal rotation onto the standard slice.
                   Default False. Only switch on when slices come from
                   SEPARATELY-TRAINED embeddings with arbitrary orientation
                   (e.g. per-slice word2vec). With a frozen encoder (E5) or
                   shared-vocabulary PPMI rows + a uniform basis, all slices
                   are ALREADY co-framed; rotating per-slice is unnecessary
                   and, on the diachronic axis, rotates away the drift NSI is
                   meant to measure.
    center       : global mean-centring (anisotropy / "cone effect" fix).
                   Recommended for E5 contextual embeddings, off for PPMI.
    """
    def src_matrix():
        if basis_source == "standard":
            return _matrix(slices[standard_key], vocab)[0]
        return np.vstack([_matrix(sl, vocab)[0] for sl in slices.values()])

    src = src_matrix()
    mu = src.mean(axis=0, keepdims=True)
    k = min(dim, min(src.shape) - 1)
    _, _, Vt = np.linalg.svd(src - mu, full_matrices=False)
    B = Vt[:k].T                                  # shared basis, applied to all

    reduced: dict[tuple, dict[str, np.ndarray]] = {}
    for key, sl in slices.items():
        M, words = _matrix(sl, vocab)
        R = (M - mu) @ B
        reduced[key] = {w: R[i] for i, w in enumerate(words)}

    if procrustes:
        std_map = reduced[standard_key]
        for key, wmap in reduced.items():
            if key == standard_key:
                continue
            shared = [w for w in wmap if w in std_map]
            if len(shared) >= 2:
                A = np.vstack([wmap[w] for w in shared])
                Bm = np.vstack([std_map[w] for w in shared])
                Rot, _ = orthogonal_procrustes(A, Bm)
                reduced[key] = {w: v @ Rot for w, v in wmap.items()}

    if center:
        allv = np.vstack([v for wmap in reduced.values() for v in wmap.values()])
        gmu = allv.mean(axis=0)
        for wmap in reduced.values():
            for w in wmap:
                wmap[w] = wmap[w] - gmu
    return reduced


# --------------------------------------------------------------------------- #
# Phase 4 — Comparative: NSI, SAI, TDI
# --------------------------------------------------------------------------- #
def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def logf(n: int) -> float:
    return math.log(1.0 + n)


def angular_displacement(v_t: np.ndarray, v_tp: np.ndarray) -> float:
    """delta in [0,1]; 0 = preserved direction, 1 = inversion."""
    return (1.0 - _cos(v_t, v_tp)) / 2.0


def nsi_sector(reduced, slices, carrier, t, tp):
    key_t, key_tp = (carrier, t), (carrier, tp)
    wt, wtp = reduced[key_t], reduced[key_tp]
    freqs_t = slices[key_t].freqs
    shared = [w for w in wt if w in wtp]
    num = den = 0.0
    contribs = []
    for w in shared:
        f = logf(freqs_t.get(w, 0))
        d = angular_displacement(wt[w], wtp[w])
        num += f * d
        den += f
        contribs.append((w, d, freqs_t.get(w, 0)))
    nsi = num / den if den else float("nan")
    contribs.sort(key=lambda x: x[1] * logf(x[2]), reverse=True)
    return nsi, contribs


def nsi_state(reduced, slices, t, tp):
    num = den = 0.0
    per_sector = {}
    for c in CARRIERS:
        if (c, t) in slices and (c, tp) in slices:
            nsi, _ = nsi_sector(reduced, slices, c, t, tp)
            ndocs = slices[(c, t)].n_docs
            w = logf(ndocs)
            num += w * nsi
            den += w
            per_sector[c] = nsi
    return (num / den if den else float("nan")), per_sector


def sai(reduced, slices, sector, t):
    """Sector Alignment Index of `sector` vs government g at time t, in [-1,1]."""
    ks, kg = (sector, t), ("g", t)
    if ks not in reduced or kg not in reduced:
        return float("nan")
    ws, wg = reduced[ks], reduced[kg]
    freqs_s = slices[ks].freqs
    shared = [w for w in ws if w in wg]
    num = den = 0.0
    for w in shared:
        f = logf(freqs_s.get(w, 0))
        num += f * _cos(ws[w], wg[w])
        den += f
    return num / den if den else float("nan")


def tdi(reduced, slices, years):
    """Per-sector SAI trajectories + composite. years = ordered list."""
    traj = {}
    for c in [c for c in CARRIERS if c != "g"]:
        traj[c] = {y: sai(reduced, slices, c, y) for y in years}
    composite = {}
    for y in years:
        vals = [traj[c][y] for c in traj if not math.isnan(traj[c][y])]
        composite[y] = sum(vals) / len(vals) if vals else float("nan")
    return traj, composite


def cluster_shifts(reduced, slices, carrier, t, tp, n_clusters=4, min_freq=2,
                   stop: set | None = None):
    """Cohesion (kappa) of hierarchical clusters on the standard space, with
    mean displacement per cluster — supports interpretation of WHAT moved."""
    stop = stop or set()
    key_t, key_tp = (carrier, t), (carrier, tp)
    wt, wtp = reduced[key_t], reduced[key_tp]
    freqs_t = slices[key_t].freqs
    words = [w for w in wt if w in wtp and freqs_t.get(w, 0) >= min_freq
             and not w.startswith("##") and len(w) > 2 and w not in stop]
    if len(words) < n_clusters + 1:
        return []
    X = np.vstack([wt[w] for w in words])
    labels = AgglomerativeClustering(n_clusters=n_clusters, linkage="average").fit_predict(X)
    out = []
    for k in range(n_clusters):
        members = [words[i] for i in range(len(words)) if labels[i] == k]
        if len(members) < 2:
            continue
        sims = [_cos(wt[a], wt[b]) for a, b in combinations(members, 2)]
        kappa = float(np.mean(sims))
        disp = float(np.mean([angular_displacement(wt[w], wtp[w]) for w in members]))
        out.append({"members": members, "kappa": round(kappa, 3), "mean_delta": round(disp, 3)})
    out.sort(key=lambda d: d["mean_delta"], reverse=True)
    return out


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run(docs: list[dict], backend: str = "cooc", dim: int = 50,
        standard_carrier: str = "g", standard_year: int | None = None,
        vocab_min_count: int = 2, basis_source: str = "pooled",
        procrustes: bool = False):
    docs, dedup_report = deduplicate(docs)
    tok = train_tokenizer(docs)
    docs = tokenize_docs(docs, tok)

    # shared vocabulary (drop very rare + structural tokens)
    counter: Counter = Counter()
    for d in docs:
        counter.update(d["_tokens"])
    vocab = [w for w, c in counter.items()
             if c >= vocab_min_count and w not in ("[UNK]", "[PAD]")]

    years = sorted({d["year"] for d in docs})
    if standard_year is None:
        standard_year = years[0]
    standard_key = (standard_carrier, standard_year)

    slices = BACKENDS[backend](docs, vocab, dim=dim)
    reduced = project_and_align(
        slices, vocab, standard_key, dim=dim, basis_source=basis_source,
        procrustes=procrustes, center=(backend == "e5"))

    t, tp = years[0], years[-1]
    state_nsi, sector_nsi = nsi_state(reduced, slices, t, tp)
    sai_table = {(c, y): sai(reduced, slices, c, y)
                 for c in CARRIERS if c != "g" for y in years}
    traj, composite = tdi(reduced, slices, years)

    return {
        "dedup": dedup_report,
        "vocab_size": len(vocab),
        "tokenizer_vocab": tok.get_vocab_size(),
        "years": years,
        "slices": {f"{c}-{y}": slices[(c, y)].n_docs
                   for c in CARRIERS for y in years if (c, y) in slices},
        "standard_key": f"{standard_carrier}-{standard_year}",
        "state_nsi": state_nsi,
        "sector_nsi": sector_nsi,
        "sai": sai_table,
        "tdi_trajectories": traj,
        "tdi_composite": composite,
        "_objs": (reduced, slices, t, tp),
    }


# --------------------------------------------------------------------------- #
# Significance + robustness utilities (added in review revision)
# --------------------------------------------------------------------------- #
def _sai_single_year(docs, year, sector, dim, mincount, standard_carrier="g"):
    """Compute SAI(sector vs government) within one year for a given doc set."""
    from collections import Counter as _C
    d, _ = deduplicate(docs)
    tok = train_tokenizer(d)
    d = tokenize_docs(d, tok)
    c = _C()
    for x in d:
        c.update(x["_tokens"])
    vocab = [w for w, n in c.items() if n >= mincount and w not in ("[UNK]", "[PAD]")]
    sl = BACKENDS["cooc"](d, vocab, dim=dim)
    if (sector, year) not in sl or (standard_carrier, year) not in sl:
        return float("nan")
    red = project_and_align(sl, vocab, (standard_carrier, year), dim=dim,
                            basis_source="pooled", procrustes=False)
    ws, wg = red[(sector, year)], red[(standard_carrier, year)]
    fs = sl[(sector, year)].freqs
    shared = [w for w in ws if w in wg]
    num = den = 0.0
    for w in shared:
        f = logf(fs.get(w, 0))
        num += f * _cos(ws[w], wg[w])
        den += f
    return num / den if den else float("nan")


def permutation_test_sai(docs, year, dim=20, mincount=2, n_perm=300, seed=0,
                         standard_carrier="g"):
    """Permutation null for the SAI of each non-government carrier.

    The government documents are held fixed; the non-government documents are
    randomly re-partitioned into the same carrier sizes. Returns, per carrier,
    the observed SAI, the null mean/sd/percentiles, and a two-sided-ish
    one-tailed p-value P(null >= observed). Also returns the probability that a
    random partition reproduces the observed ordering of the three carriers.
    """
    rng = random.Random(seed)
    gov = [d for d in docs if d["carrier"] == standard_carrier]
    nong = [d for d in docs if d["carrier"] != standard_carrier]
    sectors = sorted({d["carrier"] for d in nong})
    sizes = {s: sum(1 for d in nong if d["carrier"] == s) for s in sectors}

    observed = {s: _sai_single_year(docs, year, s, dim, mincount, standard_carrier)
                for s in sectors}

    null = {s: [] for s in sectors}
    obs_order = [s for s in sorted(sectors, key=lambda s: -observed[s])]
    order_hits = 0
    for _ in range(n_perm):
        shuffled = nong[:]
        rng.shuffle(shuffled)
        perm, i = [], 0
        for s in sectors:
            for _k in range(sizes[s]):
                dd = dict(shuffled[i]); dd["carrier"] = s; perm.append(dd); i += 1
        vals = {s: _sai_single_year(gov + perm, year, s, dim, mincount, standard_carrier)
                for s in sectors}
        for s in sectors:
            if not math.isnan(vals[s]):
                null[s].append(vals[s])
        perm_order = [s for s in sorted(sectors, key=lambda s: -vals[s])]
        if perm_order == obs_order:
            order_hits += 1

    out = {}
    for s in sectors:
        arr = np.array(null[s]) if null[s] else np.array([float("nan")])
        p = float(np.mean(arr >= observed[s])) if not math.isnan(observed[s]) else float("nan")
        out[s] = {
            "observed": float(observed[s]),
            "null_mean": float(np.nanmean(arr)),
            "null_sd": float(np.nanstd(arr)),
            "null_p05": float(np.nanpercentile(arr, 5)),
            "null_p95": float(np.nanpercentile(arr, 95)),
            "p_value": p,
        }
    out["_order_observed"] = obs_order
    out["_order_reproduced_freq"] = order_hits / n_perm
    return out


def sensitivity_grid_sai(docs, year, dims=(8, 12, 20, 30), mincounts=(2, 3),
                         standard_carrier="g"):
    """SAI of each non-government carrier across a small hyperparameter grid."""
    nong = sorted({d["carrier"] for d in docs if d["carrier"] != standard_carrier})
    rows = []
    for dim in dims:
        for mc in mincounts:
            vals = {s: _sai_single_year(docs, year, s, dim, mc, standard_carrier)
                    for s in nong}
            order = ">".join(sorted(nong, key=lambda s: -(vals[s] if not math.isnan(vals[s]) else -9)))
            rows.append({"dim": dim, "min_count": mc,
                         "sai": {s: round(vals[s], 3) for s in nong}, "order": order})
    return rows
