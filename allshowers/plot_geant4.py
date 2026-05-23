"""
shower_observables_energy_bins.py
----------------------------------
Plots shower observables (energy only, no time) split by 5-GeV energy bins
derived from the HDF5 "energies" dataset (values in MeV, range ~1–100 GeV).

Bins: [1,6), [6,11), ..., [96,101) GeV  →  20 bins

Multiprocessing:
  Each worker receives (path, idx_start, idx_end), opens the file once,
  reads its slice of showers/pdg/energies, computes all observables, closes.
  --chunk-size 5000 + 170k showers → 34 chunks, ≤32 concurrent file handles.

Usage:
    python shower_observables_energy_bins.py \
        --ml        /path/to/ml_samples.h5 \
        --sim       /path/to/sim_test.h5 \
        --out-dir   ./plots \
        --chunk-size 5000 \
        --workers   8
"""

import argparse
import os
import time
from multiprocessing import Pool

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import binned_statistic

# ------------------------------------------------------------------ constants

CLASS_NAMES = {
    0: r"$PbF2$",
    1: r"$PbOW4$",
}
NUM_LAYERS  = 76
THRESHOLD   = 1e-4        # GeV (= 0.1 MeV)
ENERGY_STEP = 5_000.0     # MeV (= 5 GeV)
ENERGY_MIN  = 1_000.0     # MeV (= 1 GeV)
ENERGY_MAX  = 100_000.0   # MeV (= 100 GeV)
SIM_COLOR   = "black"
ML_COLOR    = "steelblue"

# 20 bins: edges [1000, 6000, 11000, ..., 101000] MeV
ENERGY_EDGES  = np.arange(ENERGY_MIN, ENERGY_MAX + ENERGY_STEP + 1, ENERGY_STEP)
ENERGY_LABELS = [
    f"{int(lo/1e3)}–{int(hi/1e3)} GeV"
    for lo, hi in zip(ENERGY_EDGES[:-1], ENERGY_EDGES[1:])
]
N_EBINS = len(ENERGY_LABELS)   # 20

# ------------------------------------------------------------------ worker

def _process_chunk(args):
    """
    Open HDF5 once, read slice [idx_start:idx_end], compute observables, close.

    Per-hit arrays (cell_e, cog_dist, cog_e) carry a `local_sid` that is the
    position of the shower within the returned chunk arrays (0-based).  After
    merging chunks the main process builds a global `merged_sid` by offsetting
    each chunk's local_sid by the cumulative shower count — so merged_sid is
    always a position into the merged energies/pdg/long arrays, never a raw
    HDF5 row index.  This makes energy-bin masking straightforward.
    """
    path, idx_start, idx_end = args

    with h5py.File(path, "r") as f:
        pdg_c      = f["pdg"][idx_start:idx_end].squeeze()      # (n,)
        energies_c = f["energies"][idx_start:idx_end].squeeze()  # (n,)  MeV
        raw_c      = f["showers"][idx_start:idx_end]
        shape      = f["shape"][:]

    max_pts = int(shape[1])
    ncols   = int(shape[2])
    n       = idx_end - idx_start

    # dense tensor  (n, max_pts, ncols)
    pts = np.zeros((n, max_pts, ncols), dtype=np.float32)
    for i, flat in enumerate(raw_c):
        arr = np.asarray(flat, dtype=np.float32).reshape(-1, ncols)
        pts[i, :len(arr)] = arr

    # threshold energy column (col 3)
    pts[..., 3] = np.where(pts[..., 3] >= THRESHOLD, pts[..., 3], 0.0)

    # --- longitudinal profile  (n, NUM_LAYERS) ---
    long = np.zeros((n, NUM_LAYERS), dtype=np.float64)
    layer_idx = np.clip((pts[..., 2] + 0.1).astype(np.int32), 0, NUM_LAYERS - 1)
    np.add.at(long,
              (np.arange(n).reshape(-1, 1).repeat(max_pts, axis=1), layer_idx),
              pts[..., 3].astype(np.float64))

    # --- per-hit arrays with local shower index ---
    ce_e_list   = []   # cell energy
    ce_sid_list = []   # local shower index for each hit

    rd_dist_list = []  # CoG-centred radial distance
    rd_e_list    = []  # hit energy
    rd_sid_list  = []  # local shower index for each hit

    for i in range(n):
        e = pts[i, :, 3].astype(np.float64)
        mask = e > 0
        if mask.sum() == 0:
            continue

        e_h = e[mask]
        x_h = pts[i, :, 0][mask].astype(np.float64)
        y_h = pts[i, :, 1][mask].astype(np.float64)

        # cell energies
        ce_e_list.append(e_h)
        ce_sid_list.append(np.full(mask.sum(), i, dtype=np.int32))

        # radial distances (CoG-centred)
        e_sum = e_h.sum()
        xc = (x_h * e_h).sum() / e_sum
        yc = (y_h * e_h).sum() / e_sum
        dist = np.sqrt((x_h - xc)**2 + (y_h - yc)**2)
        rd_dist_list.append(dist)
        rd_e_list.append(e_h)
        rd_sid_list.append(np.full(mask.sum(), i, dtype=np.int32))

    def _cat(lst):
        return np.concatenate(lst) if lst else np.array([], dtype=np.float64)

    return dict(
        energies = energies_c,          # (n,)  MeV  — used for bin masking
        pdg      = pdg_c,               # (n,)
        long     = long,                # (n, NUM_LAYERS)
        ce_e     = _cat(ce_e_list),     # (H,)  GeV
        ce_sid   = _cat(ce_sid_list).astype(np.int32),   # (H,) local sid
        rd_dist  = _cat(rd_dist_list),  # (H,)
        rd_e     = _cat(rd_e_list),     # (H,)
        rd_sid   = _cat(rd_sid_list).astype(np.int32),   # (H,)
        n        = n,
    )


def _merge_chunks(results):
    """
    Concatenate chunk results.  Offset local shower indices (sid) by the
    cumulative shower count so they become positions into the merged arrays.
    """
    offset = 0
    energies_parts = []
    pdg_parts      = []
    long_parts     = []
    ce_e_parts     = []
    ce_sid_parts   = []
    rd_dist_parts  = []
    rd_e_parts     = []
    rd_sid_parts   = []

    for r in results:
        energies_parts.append(r["energies"])
        pdg_parts.append(r["pdg"])
        long_parts.append(r["long"])
        ce_e_parts.append(r["ce_e"])
        rd_dist_parts.append(r["rd_dist"])
        rd_e_parts.append(r["rd_e"])
        # offset the local sids so they index into the merged shower axis
        ce_sid_parts.append(r["ce_sid"] + offset)
        rd_sid_parts.append(r["rd_sid"] + offset)
        offset += r["n"]

    return dict(
        energies = np.concatenate(energies_parts),
        pdg      = np.concatenate(pdg_parts),
        long     = np.concatenate(long_parts, axis=0),
        ce_e     = np.concatenate(ce_e_parts),
        ce_sid   = np.concatenate(ce_sid_parts).astype(np.int32),
        rd_dist  = np.concatenate(rd_dist_parts),
        rd_e     = np.concatenate(rd_e_parts),
        rd_sid   = np.concatenate(rd_sid_parts).astype(np.int32),
    )


def _load_parallel(path, chunk_size, n_workers):
    with h5py.File(path, "r") as f:
        total = int(f["shape"][0])
    chunks = [(path, i, min(i + chunk_size, total))
              for i in range(0, total, chunk_size)]
    print(f"    {total} showers  |  {len(chunks)} chunks  |  {n_workers} workers")
    t0 = time.time()
    with Pool(n_workers) as pool:
        results = pool.map(_process_chunk, chunks)
    print(f"    merged in {time.time()-t0:.1f}s")
    return _merge_chunks(results)


# ------------------------------------------------------------------ plotting helpers

def _radial_profile(dist, e, n_events, num_bins=35, r_max=400.0):
    if len(dist) == 0 or n_events == 0:
        return None, None, None
    mean,  edges, _ = binned_statistic(dist, e, bins=num_bins, statistic="mean",  range=(0, r_max))
    std,   _,     _ = binned_statistic(dist, e, bins=num_bins, statistic="std",   range=(0, r_max))
    count, _,     _ = binned_statistic(dist, e, bins=num_bins, statistic="count", range=(0, r_max))
    mean_s = mean * count / n_events
    sem_s  = (std / np.sqrt(count.clip(min=1))) * (count / n_events)
    return edges, mean_s, sem_s


def _add_ratio_panel(fig, gs_cell, x, sim_vals, ml_vals,
                     sim_err=None, ml_err=None, xscale="linear", xlabel=""):
    inner    = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_cell,
                                               height_ratios=[3, 1], hspace=0.08)
    ax_main  = fig.add_subplot(inner[0])
    ax_ratio = fig.add_subplot(inner[1], sharex=ax_main)
    plt.setp(ax_main.get_xticklabels(), visible=False)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(sim_vals > 0, ml_vals / sim_vals, np.nan)

    ax_ratio.axhline(1.0, color="gray", lw=0.8, ls="--")
    ax_ratio.plot(x, ratio, color=ML_COLOR, lw=1.2,
                  drawstyle="steps-mid" if xscale == "linear" else "default")

    if sim_err is not None and ml_err is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_err = ratio * np.sqrt(
                np.where(sim_vals > 0, (sim_err / sim_vals)**2, 0) +
                np.where(ml_vals  > 0, (ml_err  / ml_vals )**2, 0)
            )
        ax_ratio.fill_between(x, ratio - ratio_err, ratio + ratio_err,
                              alpha=0.25, color=ML_COLOR,
                              step="mid" if xscale == "linear" else None)

    ax_ratio.set_ylim(0.5, 1.5)
    ax_ratio.set_yticks([0.5, 1.0, 1.5])
    ax_ratio.set_ylabel("ML / Sim", fontsize=7)
    ax_ratio.tick_params(labelsize=7)
    ax_ratio.grid(False)
    ax_ratio.set_xlabel(xlabel, fontsize=8)
    if xscale == "log":
        ax_ratio.set_xscale("log")
    return ax_main, ax_ratio


# ------------------------------------------------------------------ per-bin plot

ROW_CONFIGS = [
    ("All",                        None),
    (f"Class 0: {CLASS_NAMES[0]}", 0),
    (f"Class 1: {CLASS_NAMES[1]}", 1),
]
NROWS = len(ROW_CONFIGS)   # 3
NCOLS = 3                  # Longitudinal | Cell Energy | Radial


def _compute_row_data(ebin_idx, class_val, s_data, m_data):
    """
    Select, cap, and compute all observables for one (energy bin, class) row.
    Returns a dict ready for plotting, or None if no showers pass the cuts.
    """
    e_lo = ENERGY_EDGES[ebin_idx]
    e_hi = ENERGY_EDGES[ebin_idx + 1]

    s_emask = (s_data["energies"] >= e_lo) & (s_data["energies"] < e_hi)
    m_emask = (m_data["energies"] >= e_lo) & (m_data["energies"] < e_hi)
    if class_val is not None:
        s_emask &= (s_data["pdg"] == class_val)
        m_emask &= (m_data["pdg"] == class_val)

    s_idx = np.where(s_emask)[0]
    m_idx = np.where(m_emask)[0]

    # All:     5k sim + 5k ML = 10k total across both files
    # Class 0 / Class 1: up to 10k sim + 10k ML independently
    MAX_N = 5_000 if class_val is None else 10_000
    n = min(len(s_idx), len(m_idx), MAX_N)
    if n == 0:
        return None

    rng   = np.random.default_rng(42)
    s_idx = rng.choice(s_idx, size=n, replace=False)
    m_idx = rng.choice(m_idx, size=n, replace=False)

    s_bool = np.zeros(len(s_data["energies"]), dtype=bool)
    m_bool = np.zeros(len(m_data["energies"]), dtype=bool)
    s_bool[s_idx] = True
    m_bool[m_idx] = True

    # longitudinal
    sl      = s_data["long"][s_idx]
    ml_l    = m_data["long"][m_idx]
    sl_mean = sl.mean(0);    sl_sem = sl.std(0) / np.sqrt(n)
    ml_mean = ml_l.mean(0); ml_sem = ml_l.std(0) / np.sqrt(n)

    # cell energies
    s_ce = s_data["ce_e"][s_bool[s_data["ce_sid"]]]
    m_ce = m_data["ce_e"][m_bool[m_data["ce_sid"]]]

    # radial
    s_rd = s_data["rd_dist"][s_bool[s_data["rd_sid"]]]
    s_re = s_data["rd_e"][s_bool[s_data["rd_sid"]]]
    m_rd = m_data["rd_dist"][m_bool[m_data["rd_sid"]]]
    m_re = m_data["rd_e"][m_bool[m_data["rd_sid"]]]

    all_r = np.concatenate([s_rd, m_rd]) if len(s_rd) + len(m_rd) > 0 else np.array([1.0])
    R_MAX = float(np.percentile(all_r, 99)) if len(all_r) > 0 else 400.0

    edges_r, sr_mean, sr_sem = _radial_profile(s_rd, s_re, n_events=n, r_max=R_MAX)
    _,        mr_mean, mr_sem = _radial_profile(m_rd, m_re, n_events=n, r_max=R_MAX)
    if sr_mean is not None:
        sr_mean *= 1e3;  sr_sem *= 1e3   # GeV → MeV
        mr_mean *= 1e3;  mr_sem *= 1e3

    return dict(n=n, sl_mean=sl_mean, sl_sem=sl_sem, ml_mean=ml_mean, ml_sem=ml_sem,
                s_ce=s_ce, m_ce=m_ce,
                edges_r=edges_r, sr_mean=sr_mean, sr_sem=sr_sem,
                mr_mean=mr_mean, mr_sem=mr_sem, R_MAX=R_MAX)


def _fill_row(fig, outer_gs, row_i, row_label, rd):
    """
    Fill one row (3 panels) of the figure from pre-computed row data `rd`.
    If rd is None, write a centred "No data" message across the row.
    """
    layers = np.arange(1, NUM_LAYERS + 1)

    if rd is None:
        for col in range(NCOLS):
            ax = fig.add_subplot(outer_gs[row_i, col])
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
        # row label above the row (col 0 axes fraction, above top)
        ax0 = fig.add_subplot(outer_gs[row_i, 0])
        ax0.annotate(row_label, xy=(0, 1.22), xycoords="axes fraction",
                     fontsize=10, fontweight="bold", color="#111111", va="bottom", ha="left")
        return

    n = rd["n"]

    # ---- row label (left margin of col 0 main panel) ----
    # We annotate after creating the first axis below.

    # ---- Col 0: Longitudinal Energy ----
    ax_main, ax_ratio = _add_ratio_panel(
        fig, outer_gs[row_i, 0],
        layers, rd["sl_mean"], rd["ml_mean"], rd["sl_sem"], rd["ml_sem"],
        xlabel="Plane")
    ax_main.plot(layers, rd["sl_mean"], color=SIM_COLOR, lw=1.5,
                 drawstyle="steps-mid", label=f"Simulated ({n})")
    ax_main.fill_between(layers, rd["sl_mean"] - rd["sl_sem"],
                         rd["sl_mean"] + rd["sl_sem"],
                         alpha=0.15, color=SIM_COLOR, step="mid")
    ax_main.plot(layers, rd["ml_mean"], color=ML_COLOR, lw=1.5,
                 drawstyle="steps-mid", label=f"ML ({n})")
    ax_main.fill_between(layers, rd["ml_mean"] - rd["ml_sem"],
                         rd["ml_mean"] + rd["ml_sem"],
                         alpha=0.20, color=ML_COLOR, step="mid")
    ax_main.set_ylabel("Mean Energy [GeV]", fontsize=8)
    ax_main.set_xticks(np.arange(1, NUM_LAYERS + 1, 10))
    ax_main.grid(False); ax_main.legend(fontsize=7)
    if row_i == 0:
        ax_main.set_title("Longitudinal Energy Profile", fontsize=9)
    # row label above the row, left-aligned above col 0 — matches reference style
    ax_main.annotate(row_label, xy=(0, 1.22), xycoords="axes fraction",
                     fontsize=10, fontweight="bold", color="#111111", va="bottom", ha="left")

    # ---- Col 1: Cell Energy Spectrum ----
    s_ce, m_ce = rd["s_ce"], rd["m_ce"]
    if len(s_ce) > 0 and len(m_ce) > 0:
        e_min_plot = max(min(s_ce.min(), m_ce.min()), 1e-4)
        e_max_plot = max(s_ce.max(), m_ce.max())
        bins  = np.logspace(np.log10(e_min_plot), np.log10(e_max_plot), 80)
        s_cnt, _ = np.histogram(s_ce, bins=bins)
        m_cnt, _ = np.histogram(m_ce, bins=bins)
        bin_c     = np.sqrt(bins[:-1] * bins[1:])

        inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_gs[row_i, 1],
                                                  height_ratios=[3, 1], hspace=0.08)
        ax_m = fig.add_subplot(inner[0])
        ax_r = fig.add_subplot(inner[1], sharex=ax_m)
        plt.setp(ax_m.get_xticklabels(), visible=False)

        ax_m.stairs(s_cnt, bins, color=SIM_COLOR, lw=1.5, label=f"Simulated ({n})")
        ax_m.errorbar(bin_c, s_cnt, yerr=np.sqrt(s_cnt),
                      fmt="none", color=SIM_COLOR, lw=0.8, capsize=2)
        ax_m.stairs(m_cnt, bins, color=ML_COLOR, lw=1.5, label=f"ML ({n})")
        ax_m.errorbar(bin_c, m_cnt, yerr=np.sqrt(m_cnt),
                      fmt="none", color=ML_COLOR, lw=0.8, capsize=2)
        ax_m.set_yscale("log"); ax_m.set_xscale("log")
        ax_m.set_xlim(e_min_plot, e_max_plot)
        ax_m.set_ylabel("Number of cells", fontsize=8)
        ax_m.grid(False); ax_m.legend(fontsize=7)
        if row_i == 0:
            ax_m.set_title("Cell Energy Spectrum", fontsize=9)

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(s_cnt > 0, m_cnt / s_cnt, np.nan)
            r_err = np.where(s_cnt > 0,
                             ratio * np.sqrt(1.0 / np.where(m_cnt > 0, m_cnt, 1)
                                           + 1.0 / np.where(s_cnt > 0, s_cnt, 1)),
                             np.nan)
        ax_r.axhline(1.0, color="gray", lw=0.8, ls="--")
        ax_r.plot(bin_c, ratio, color=ML_COLOR, lw=1.2)
        ax_r.fill_between(bin_c, ratio - r_err, ratio + r_err, alpha=0.25, color=ML_COLOR)
        ax_r.set_xscale("log"); ax_r.set_ylim(0.5, 1.5); ax_r.set_yticks([0.5, 1.0, 1.5])
        ax_r.set_ylabel("ML / Sim", fontsize=7)
        ax_r.set_xlabel("Cell energy [GeV]", fontsize=8)
        ax_r.tick_params(labelsize=7); ax_r.grid(False)
    else:
        fig.add_subplot(outer_gs[row_i, 1]).text(0.5, 0.5, "No hits",
                                                  ha="center", va="center")

    # ---- Col 2: Radial Energy Profile ----
    if rd["sr_mean"] is not None:
        edges_r  = rd["edges_r"]
        bin_c_r  = 0.5 * (edges_r[:-1] + edges_r[1:])
        ax_main, ax_ratio = _add_ratio_panel(
            fig, outer_gs[row_i, 2],
            bin_c_r, rd["sr_mean"], rd["mr_mean"], rd["sr_sem"], rd["mr_sem"],
            xlabel="Radial Distance [m]")
        ax_main.stairs(rd["sr_mean"], edges_r, color=SIM_COLOR, lw=1.5,
                       label=f"Simulated ({n})")
        ax_main.stairs(rd["sr_mean"] + rd["sr_sem"], edges_r,
                       baseline=rd["sr_mean"] - rd["sr_sem"],
                       color=SIM_COLOR, alpha=0.2, fill=True)
        ax_main.stairs(rd["mr_mean"], edges_r, color=ML_COLOR, lw=1.5,
                       label=f"ML ({n})")
        ax_main.stairs(rd["mr_mean"] + rd["mr_sem"], edges_r,
                       baseline=rd["mr_mean"] - rd["mr_sem"],
                       color=ML_COLOR, alpha=0.2, fill=True)
        ax_main.set_yscale("log")
        ax_main.set_ylabel("Mean Energy [MeV]", fontsize=8)
        ax_main.set_xlim(0, rd["R_MAX"])
        ax_main.grid(False); ax_main.legend(fontsize=7)
        if row_i == 0:
            ax_main.set_title("Radial Energy Profile", fontsize=9)
        ax_ratio.set_ylim(0.5, 1.5); ax_ratio.set_yticks([0.5, 1.0, 1.5])
    else:
        fig.add_subplot(outer_gs[row_i, 2]).text(0.5, 0.5, "No radial data",
                                                  ha="center", va="center")


def _plot_energy_bin(ebin_idx, ebin_label, s_data, m_data, out_dir):
    """
    Produce one figure with NROWS=3 rows (All / Class 0 / Class 1) and
    NCOLS=3 columns (Longitudinal | Cell Energy | Radial) for one energy bin.
    """
    # compute observables for every row up front
    row_data = [
        _compute_row_data(ebin_idx, class_val, s_data, m_data)
        for _, class_val in ROW_CONFIGS
    ]

    fig = plt.figure(figsize=(NCOLS * 4.4, NROWS * 4.8))
    outer = gridspec.GridSpec(
        NROWS, NCOLS, figure=fig,
        hspace=0.55, wspace=0.42,
        top=0.93, bottom=0.05, left=0.08, right=0.98,
    )
    fig.suptitle(f"Incident Energies: {ebin_label}", fontsize=14, fontweight="bold", y=0.99)

    for row_i, ((row_label, _), rd) in enumerate(zip(ROW_CONFIGS, row_data)):
        _fill_row(fig, outer, row_i, row_label, rd)

    safe  = ebin_label.replace(" ", "").replace("–", "-")
    fname = os.path.join(out_dir, f"shower_{safe}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fname


# ------------------------------------------------------------------ main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ml",         required=True)
    parser.add_argument("--sim",        required=True)
    parser.add_argument("--out-dir",    default="plots")
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--workers",    type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading Simulated …")
    s_data = _load_parallel(args.sim, args.chunk_size, args.workers)
    print("Loading ML …")
    m_data = _load_parallel(args.ml,  args.chunk_size, args.workers)

    print(f"\nSimulated: {len(s_data['energies'])} showers  "
          f"energies {s_data['energies'].min():.0f}–{s_data['energies'].max():.0f} MeV")
    print(f"ML:        {len(m_data['energies'])} showers  "
          f"energies {m_data['energies'].min():.0f}–{m_data['energies'].max():.0f} MeV")
    print(f"Bins: {N_EBINS}  ({ENERGY_LABELS[0]} … {ENERGY_LABELS[-1]})\n")

    t0 = time.time()
    for ebin_idx, ebin_label in enumerate(ENERGY_LABELS):
        fname = _plot_energy_bin(ebin_idx, ebin_label, s_data, m_data, args.out_dir)
        print(f"  [{ebin_idx+1:2d}/{N_EBINS}]  {os.path.basename(fname)}")

    print(f"\nDone in {time.time()-t0:.1f}s  →  {args.out_dir}/")


if __name__ == "__main__":
    main()