"""
plot_corsika_electron.py
========================
Compare one CORSIKA-simulated file against 2–3 ML-generated files.

Configure the file paths and labels in the USER CONFIG section below.
Each ML entry is a dict with keys:
  path  : str   – path to the .h5 file
  label : str   – short label used in legends (e.g. "ML-v1", "ML-v2")
  color : str   – matplotlib colour string
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ===========================================================================
# USER CONFIG – edit this section
# ===========================================================================

SIMULATED_FILE = (
    "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/"
    "tambo_simulations_for_training/h5_files_v3/combined_electrons_test.h5"
)

# Add 2 or 3 ML entries; remove or comment-out entries you don't need.
ML_FILES = [
    {
        "path":  "/n/home04/hhanif/AllShowers/results/20260610_003211_Electron-Allshower/samples00.h5",
        "label": "ML-EMA",
        "color": "steelblue",
    },
    {
        "path":  "/n/home04/hhanif/AllShowers/results/20260610_003211_Electron-Allshower/samples01.h5",
        "label": "ML-Best",
        "color": "darkorange",
    },

]

OUTPUT_PDF = (
    "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/"
    "tambo_simulations_for_training/h5_files_v3/"
    "shower_observables_reference_style_electron.pdf"
)

# ===========================================================================
# CONSTANTS
# ===========================================================================

CLASS_NAMES = {
    0: r"$e^\pm$/$\gamma$/$\pi^0$",
    1: r"$\pi^\pm$",
}
NUM_LAYERS = 24
US         = 1e6        # seconds → microseconds
THRESHOLD  = 1e-4       # GeV (= 0.1 MeV)
SIM_COLOR  = "black"

# ===========================================================================
# LOADERS
# ===========================================================================

def load_file(path):
    print(f"  Reading {path} …")
    with h5py.File(path, "r") as f:
        pdg   = f["pdg"][:]
        raw   = f["showers"][:]
        shape = f["shape"][:]

    N, max_pts, ncols = int(shape[0]), int(shape[1]), int(shape[2])
    print(f"    {N} showers, {max_pts} max pts, {ncols} cols")

    lengths  = np.array([len(r) // ncols for r in raw], dtype=np.int32)
    flat_all = np.concatenate([np.asarray(r, dtype=np.float32) for r in raw])
    hits_all = flat_all.reshape(-1, ncols)

    pts     = np.zeros((N, max_pts, ncols), dtype=np.float32)
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    for i, (start, L) in enumerate(zip(offsets[:-1], lengths)):
        if L > 0:
            pts[i, :L] = hits_all[start : start + L]

    pts[..., 3] = np.where(pts[..., 3] >= THRESHOLD, pts[..., 3], 0.0)
    return pts, pdg, ncols


# ===========================================================================
# OBSERVABLE COMPUTATIONS
# ===========================================================================

def compute_longitudinal(pts, num_layers=NUM_LAYERS):
    N = pts.shape[0]
    energy_per_layer = np.zeros((N, num_layers), dtype=np.float64)
    layer_idx  = np.clip((pts[..., 2] + 0.1).astype(np.int32), 0, num_layers - 1)
    energies   = pts[..., 3].astype(np.float64)
    shower_idx = np.arange(N).reshape(-1, 1).repeat(pts.shape[1], axis=1)
    np.add.at(energy_per_layer, (shower_idx, layer_idx), energies)
    return energy_per_layer


def compute_time_per_layer(pts, ncols, num_layers=NUM_LAYERS):
    if ncols < 5:
        return None
    N          = pts.shape[0]
    time_sum   = np.zeros((N, num_layers), dtype=np.float64)
    time_count = np.zeros((N, num_layers), dtype=np.float64)
    mask       = pts[..., 3] > 0
    layer_idx  = np.clip((pts[..., 2] + 0.1).astype(np.int32), 0, num_layers - 1)
    t          = pts[..., 4].astype(np.float64)
    for i in range(N):
        m  = mask[i]
        li = layer_idx[i][m]
        ti = t[i][m]
        np.add.at(time_sum[i],   li, ti)
        np.add.at(time_count[i], li, 1)
    return time_sum / time_count.clip(min=1)


def compute_radial_profile(pts, n_events, num_bins=35, r_max=400.0):
    from scipy.stats import binned_statistic
    all_dist, all_e = [], []
    for i in range(pts.shape[0]):
        e    = pts[i, :, 3].astype(np.float64)
        x    = pts[i, :, 0].astype(np.float64)
        y    = pts[i, :, 1].astype(np.float64)
        mask = e > 0
        if mask.sum() == 0:
            continue
        e_h   = e[mask];  x_h = x[mask];  y_h = y[mask]
        e_sum = e_h.sum()
        xc    = (x_h * e_h).sum() / e_sum
        yc    = (y_h * e_h).sum() / e_sum
        dist  = np.sqrt((x_h - xc)**2 + (y_h - yc)**2)
        all_dist.append(dist);  all_e.append(e_h)
    all_dist = np.concatenate(all_dist)
    all_e    = np.concatenate(all_e)
    mean,  edges, _ = binned_statistic(all_dist, all_e, bins=num_bins, statistic="mean",  range=(0, r_max))
    std,   _,     _ = binned_statistic(all_dist, all_e, bins=num_bins, statistic="std",   range=(0, r_max))
    count, _,     _ = binned_statistic(all_dist, all_e, bins=num_bins, statistic="count", range=(0, r_max))
    mean_shower = mean * count / n_events
    sem_shower  = (std / np.sqrt(count.clip(min=1))) * (count / n_events)
    return edges, mean_shower, sem_shower


def compute_radial_time_profile(pts, ncols, n_events, num_bins=35, r_max=400.0):
    if ncols < 5:
        return None, None, None
    from scipy.stats import binned_statistic
    all_dist, all_t = [], []
    for i in range(pts.shape[0]):
        e    = pts[i, :, 3].astype(np.float64)
        x    = pts[i, :, 0].astype(np.float64)
        y    = pts[i, :, 1].astype(np.float64)
        t    = pts[i, :, 4].astype(np.float64)
        mask = e > 0
        if mask.sum() == 0:
            continue
        e_h   = e[mask];  x_h = x[mask];  y_h = y[mask];  t_h = t[mask]
        e_sum = e_h.sum()
        xc    = (x_h * e_h).sum() / e_sum
        yc    = (y_h * e_h).sum() / e_sum
        dist  = np.sqrt((x_h - xc)**2 + (y_h - yc)**2)
        all_dist.append(dist);  all_t.append(t_h)
    all_dist = np.concatenate(all_dist)
    all_t    = np.concatenate(all_t)
    mean,  edges, _ = binned_statistic(all_dist, all_t, bins=num_bins, statistic="mean",  range=(0, r_max))
    std,   _,     _ = binned_statistic(all_dist, all_t, bins=num_bins, statistic="std",   range=(0, r_max))
    count, _,     _ = binned_statistic(all_dist, all_t, bins=num_bins, statistic="count", range=(0, r_max))
    mean_shower = mean * count / n_events
    sem_shower  = (std / np.sqrt(count.clip(min=1))) * (count / n_events)
    return edges, mean_shower, sem_shower


def _cog_distances(pts_sel):
    dists = []
    for i in range(pts_sel.shape[0]):
        e    = pts_sel[i, :, 3].astype(np.float64)
        mask = e > 0
        if mask.sum() == 0:
            continue
        e_h   = e[mask]
        x_h   = pts_sel[i, :, 0][mask].astype(np.float64)
        y_h   = pts_sel[i, :, 1][mask].astype(np.float64)
        e_sum = e_h.sum()
        xc    = (x_h * e_h).sum() / e_sum
        yc    = (y_h * e_h).sum() / e_sum
        dists.append(np.sqrt((x_h - xc)**2 + (y_h - yc)**2))
    return np.concatenate(dists) if dists else np.array([1.0])


# ===========================================================================
# HELPERS
# ===========================================================================

def mask_for(pdg_arr, pdg_val):
    return np.ones(len(pdg_arr), dtype=bool) if pdg_val is None else pdg_arr == pdg_val


def capped_indices(s_mask, m_mask, seed=42):
    s_idx = np.where(s_mask)[0]
    m_idx = np.where(m_mask)[0]
    n     = min(len(s_idx), len(m_idx))
    rng   = np.random.default_rng(seed)
    s_idx = rng.choice(s_idx, size=n, replace=False)
    m_idx = rng.choice(m_idx, size=n, replace=False)
    return s_idx, m_idx, n


def add_ratio_panel(fig, gs_cell, x, sim_vals, ml_vals_list,
                    sim_err=None, ml_err_list=None,
                    ml_colors=None, ml_labels=None,
                    xscale="linear", xlabel=""):
    """
    Creates a (main, ratio) subplot pair inside gs_cell.
    ml_vals_list : list of arrays, one per ML file.
    Returns (ax_main, ax_ratio).
    """
    inner    = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs_cell, height_ratios=[3, 1], hspace=0.08
    )
    ax_main  = fig.add_subplot(inner[0])
    ax_ratio = fig.add_subplot(inner[1], sharex=ax_main)
    plt.setp(ax_main.get_xticklabels(), visible=False)

    ax_ratio.axhline(1.0, color="gray", lw=0.8, ls="--")

    for k, ml_vals in enumerate(ml_vals_list):
        color    = ml_colors[k]  if ml_colors  else "steelblue"
        ml_err   = ml_err_list[k] if ml_err_list else None
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(sim_vals > 0, ml_vals / sim_vals, np.nan)
        ax_ratio.plot(x, ratio, color=color, lw=1.2,
                      drawstyle="steps-mid" if xscale == "linear" else "default")
        if sim_err is not None and ml_err is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio_err = ratio * np.sqrt(
                    np.where(sim_vals > 0, (sim_err  / sim_vals)**2, 0) +
                    np.where(ml_vals  > 0, (ml_err   / ml_vals )**2, 0)
                )
            ax_ratio.fill_between(x, ratio - ratio_err, ratio + ratio_err,
                                  alpha=0.20, color=color,
                                  step="mid" if xscale == "linear" else None)

    ax_ratio.set_ylim(0.5, 1.5)
    ax_ratio.set_yticks([0.95, 1.0, 1.05])
    ax_ratio.set_ylabel("ML / Sim", fontsize=7)
    ax_ratio.tick_params(labelsize=7)
    ax_ratio.grid(False)
    ax_ratio.set_xlabel(xlabel, fontsize=8)
    if xscale == "log":
        ax_ratio.set_xscale("log")

    return ax_main, ax_ratio


# ===========================================================================
# LOAD DATA
# ===========================================================================

print("Loading Simulated …")
s_pts, s_pdg, s_ncols = load_file(SIMULATED_FILE)

ml_data = []
for entry in ML_FILES:
    print(f"Loading {entry['label']} …")
    pts, pdg, ncols = load_file(entry["path"])
    ml_data.append({
        "pts":    pts,
        "pdg":    pdg,
        "ncols":  ncols,
        "label":  entry["label"],
        "color":  entry["color"],
    })

print("Pre-computing longitudinal profiles …")
s_long   = compute_longitudinal(s_pts)
s_tplane = compute_time_per_layer(s_pts, s_ncols)

for d in ml_data:
    d["long"]   = compute_longitudinal(d["pts"])
    d["tplane"] = compute_time_per_layer(d["pts"], d["ncols"])

print("Done.\n")

layers = np.arange(1, NUM_LAYERS + 1)

# ===========================================================================
# FIGURE LAYOUT
# ===========================================================================

row_configs = [
    ("All", None),
    (f"Class 0: {CLASS_NAMES[0]}", 0),
    (f"Class 1: {CLASS_NAMES[1]}", 1),
]

NROWS = len(row_configs)
NCOLS = 5

fig      = plt.figure(figsize=(NCOLS * 4.2, NROWS * 4.8))
outer_gs = gridspec.GridSpec(
    NROWS, NCOLS,
    figure=fig,
    hspace=0.55, wspace=0.38,
    top=0.96, bottom=0.05, left=0.05, right=0.99,
)

# ===========================================================================
# PLOT LOOP
# ===========================================================================

for row_i, (row_label, class_val) in enumerate(row_configs):

    s_mask = mask_for(s_pdg, class_val)

    # ---- balance sample sizes across sim + all ML files ----
    ml_masks   = [mask_for(d["pdg"], class_val) for d in ml_data]
    ml_idx_raw = [np.where(m)[0] for m in ml_masks]
    s_idx_raw  = np.where(s_mask)[0]
    n          = min(len(s_idx_raw), *[len(i) for i in ml_idx_raw])
    rng        = np.random.default_rng(42)
    s_idx      = rng.choice(s_idx_raw, size=n, replace=False)
    ml_idxs    = [rng.choice(idx, size=n, replace=False) for idx in ml_idx_raw]

    header = f"{'All' if class_val is None else row_label}  —  samples: {n}"

    ml_colors = [d["color"] for d in ml_data]

    # ---- Col 0: Longitudinal Energy Profile ----
    sl      = s_long[s_idx]
    sl_mean = sl.mean(0);  sl_sem = sl.std(0) / np.sqrt(n)

    ml_long_means = [ml_data[k]["long"][ml_idxs[k]].mean(0) for k in range(len(ml_data))]
    ml_long_sems  = [ml_data[k]["long"][ml_idxs[k]].std(0)  / np.sqrt(n) for k in range(len(ml_data))]

    ax_main, ax_ratio = add_ratio_panel(
        fig, outer_gs[row_i, 0],
        layers, sl_mean, ml_long_means, sl_sem, ml_long_sems,
        ml_colors=ml_colors, xlabel="Plane"
    )
    ax_main.plot(layers, sl_mean, color=SIM_COLOR, lw=1.5, drawstyle="steps-mid",
                 label=f"CORSIKA ({n})")
    ax_main.fill_between(layers, sl_mean - sl_sem, sl_mean + sl_sem,
                         alpha=0.15, color=SIM_COLOR, step="mid")
    for k, d in enumerate(ml_data):
        ax_main.plot(layers, ml_long_means[k], color=d["color"], lw=1.5,
                     drawstyle="steps-mid", label=f"{d['label']} ({n})")
        ax_main.fill_between(layers, ml_long_means[k] - ml_long_sems[k],
                             ml_long_means[k] + ml_long_sems[k],
                             alpha=0.20, color=d["color"], step="mid")
    ax_main.set_ylabel("Mean Energy [GeV]", fontsize=8)
    ax_main.set_xticks(np.arange(1, NUM_LAYERS + 1, 4))
    ax_main.grid(False);  ax_main.legend(fontsize=7)
    ax_main.set_title("Longitudinal Energy Profile", fontsize=9)
    ax_main.annotate(header, xy=(0, 1.22), xycoords="axes fraction",
                     fontsize=10, fontweight="bold", color="#111111")

    # ---- Col 1: Cell Energy Spectrum ----
    s_ce = s_pts[s_idx][..., 3].ravel()
    s_ce = s_ce[s_ce > 0]
    ml_ce_list = []
    for k, d in enumerate(ml_data):
        ce = d["pts"][ml_idxs[k]][..., 3].ravel()
        ml_ce_list.append(ce[ce > 0])

    e_min  = 1e-2
    e_max  = max(s_ce.max(), *[ce.max() for ce in ml_ce_list])
    bins   = np.logspace(np.log10(e_min), np.log10(e_max), 80)
    bin_centers = np.sqrt(bins[:-1] * bins[1:])

    s_counts, _ = np.histogram(s_ce, bins=bins)
    ml_counts_list = [np.histogram(ce, bins=bins)[0] for ce in ml_ce_list]

    inner    = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer_gs[row_i, 1], height_ratios=[3, 1], hspace=0.08
    )
    ax_main  = fig.add_subplot(inner[0])
    ax_ratio = fig.add_subplot(inner[1], sharex=ax_main)
    plt.setp(ax_main.get_xticklabels(), visible=False)

    ax_main.stairs(s_counts, bins, color=SIM_COLOR, lw=1.5, label=f"Simulated ({n})")
    ax_main.errorbar(bin_centers, s_counts, yerr=np.sqrt(s_counts),
                     fmt="none", color=SIM_COLOR, lw=0.8, capsize=2)
    for k, d in enumerate(ml_data):
        mc = ml_counts_list[k]
        ax_main.stairs(mc, bins, color=d["color"], lw=1.5, label=f"{d['label']} ({n})")
        ax_main.errorbar(bin_centers, mc, yerr=np.sqrt(mc),
                         fmt="none", color=d["color"], lw=0.8, capsize=2)
    ax_main.set_yscale("log");  ax_main.set_xscale("log")
    ax_main.set_xlim(e_min, e_max)
    ax_main.set_ylabel("Number of cells", fontsize=8)
    ax_main.grid(False);  ax_main.legend(fontsize=7)
    ax_main.set_title("Cell Energy Spectrum", fontsize=9)

    ax_ratio.axhline(1.0, color="gray", lw=0.8, ls="--")
    for k, d in enumerate(ml_data):
        mc = ml_counts_list[k]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(s_counts > 0, mc / s_counts, np.nan)
            rerr  = np.where(s_counts > 0,
                             ratio * np.sqrt(1.0 / np.where(mc > 0, mc, 1)
                                           + 1.0 / np.where(s_counts > 0, s_counts, 1)),
                             np.nan)
        ax_ratio.plot(bin_centers, ratio, color=d["color"], lw=1.2)
        ax_ratio.fill_between(bin_centers, ratio - rerr, ratio + rerr,
                              alpha=0.25, color=d["color"])
    ax_ratio.set_xscale("log")
    ax_ratio.set_ylim(0.5, 1.5);  ax_ratio.set_yticks([0.5, 1.0, 1.5])
    ax_ratio.set_ylabel("ML / Sim", fontsize=7)
    ax_ratio.set_xlabel("Cell energy [GeV]", fontsize=8)
    ax_ratio.tick_params(labelsize=7);  ax_ratio.grid(False)
    for ax in [ax_main, ax_ratio]:
        ax.tick_params(labelsize=7)

    # ---- Col 2: Radial Energy Profile ----
    all_r_arrays = [_cog_distances(s_pts[s_idx])] + \
                   [_cog_distances(d["pts"][ml_idxs[k]]) for k, d in enumerate(ml_data)]
    R_MAX_M = float(np.percentile(np.concatenate(all_r_arrays), 99))

    edges, sr_mean, sr_sem = compute_radial_profile(s_pts[s_idx],            n_events=n, r_max=R_MAX_M)
    ml_rad = [compute_radial_profile(d["pts"][ml_idxs[k]], n_events=n, r_max=R_MAX_M)
              for k, d in enumerate(ml_data)]

    sr_mean *= 1e3;  sr_sem *= 1e3
    ml_rad_means = [x[1] * 1e3 for x in ml_rad]
    ml_rad_sems  = [x[2] * 1e3 for x in ml_rad]
    bin_centers  = 0.5 * (edges[:-1] + edges[1:])

    ax_main, ax_ratio = add_ratio_panel(
        fig, outer_gs[row_i, 2],
        bin_centers, sr_mean, ml_rad_means, sr_sem, ml_rad_sems,
        ml_colors=ml_colors, xlabel="Radial Distance [m]"
    )
    ax_main.stairs(sr_mean, edges, color=SIM_COLOR, lw=1.5, label=f"Simulated ({n})")
    ax_main.stairs(sr_mean + sr_sem, edges, baseline=sr_mean - sr_sem,
                   color=SIM_COLOR, alpha=0.2, fill=True)
    for k, d in enumerate(ml_data):
        ax_main.stairs(ml_rad_means[k], edges, color=d["color"], lw=1.5, label=f"{d['label']} ({n})")
        ax_main.stairs(ml_rad_means[k] + ml_rad_sems[k], edges,
                       baseline=ml_rad_means[k] - ml_rad_sems[k],
                       color=d["color"], alpha=0.2, fill=True)
    ax_main.set_yscale("log")
    ax_main.set_ylabel("Mean Energy [MeV]", fontsize=8)
    ax_main.set_xlim(0, R_MAX_M)
    ax_main.grid(False);  ax_main.legend(fontsize=7)
    ax_main.set_title("Radial Energy Profile", fontsize=9)
    ax_ratio.set_ylim(0.5, 1.5);  ax_ratio.set_yticks([0.5, 1.0, 1.5])

    # ---- Col 3: Longitudinal Time Profile ----
    if s_tplane is not None and all(d["tplane"] is not None for d in ml_data):
        st      = s_tplane[s_idx]
        st_mean = st.mean(0) * US;  st_sem = st.std(0) / np.sqrt(n) * US
        ml_tp_means = [ml_data[k]["tplane"][ml_idxs[k]].mean(0) * US for k in range(len(ml_data))]
        ml_tp_sems  = [ml_data[k]["tplane"][ml_idxs[k]].std(0)  / np.sqrt(n) * US for k in range(len(ml_data))]

        ax_main, ax_ratio = add_ratio_panel(
            fig, outer_gs[row_i, 3],
            layers, st_mean, ml_tp_means, st_sem, ml_tp_sems,
            ml_colors=ml_colors, xlabel="Plane"
        )
        ax_main.plot(layers, st_mean, color=SIM_COLOR, lw=1.5, drawstyle="steps-mid",
                     label=f"Simulated ({n})")
        ax_main.fill_between(layers, st_mean - st_sem, st_mean + st_sem,
                             alpha=0.15, color=SIM_COLOR, step="mid")
        for k, d in enumerate(ml_data):
            ax_main.plot(layers, ml_tp_means[k], color=d["color"], lw=1.5,
                         drawstyle="steps-mid", label=f"{d['label']} ({n})")
            ax_main.fill_between(layers, ml_tp_means[k] - ml_tp_sems[k],
                                 ml_tp_means[k] + ml_tp_sems[k],
                                 alpha=0.20, color=d["color"], step="mid")
        ax_main.set_ylabel(r"Mean $t$ [$\mu$s]", fontsize=8)
        ax_main.set_xticks(np.arange(1, NUM_LAYERS + 1, 4))
        ax_main.grid(False);  ax_main.legend(fontsize=7)
        ax_main.set_title("Longitudinal Time Profile", fontsize=9)
        ax_ratio.set_ylim(0.5, 1.5);  ax_ratio.set_yticks([0.5, 1.0, 1.5])
    else:
        fig.add_subplot(outer_gs[row_i, 3]).text(0.5, 0.5, "No time data",
                                                  ha="center", va="center")

    # ---- Col 4: Radial Time Profile ----
    s_rtp  = compute_radial_time_profile(s_pts[s_idx], s_ncols, n_events=n, r_max=R_MAX_M)
    ml_rtp = [compute_radial_time_profile(d["pts"][ml_idxs[k]], d["ncols"],
                                           n_events=n, r_max=R_MAX_M)
              for k, d in enumerate(ml_data)]

    if s_rtp[0] is not None and all(x[0] is not None for x in ml_rtp):
        edges_t     = s_rtp[0]
        sr_t_mean   = s_rtp[1] * US;  sr_t_sem = s_rtp[2] * US
        ml_rtp_means = [x[1] * US for x in ml_rtp]
        ml_rtp_sems  = [x[2] * US for x in ml_rtp]
        t_bin_centers = 0.5 * (edges_t[:-1] + edges_t[1:])

        ax_main, ax_ratio = add_ratio_panel(
            fig, outer_gs[row_i, 4],
            t_bin_centers, sr_t_mean, ml_rtp_means, sr_t_sem, ml_rtp_sems,
            ml_colors=ml_colors, xlabel="Radial Distance [m]"
        )
        ax_main.stairs(sr_t_mean, edges_t, color=SIM_COLOR, lw=1.5, label=f"Simulated ({n})")
        ax_main.stairs(sr_t_mean + sr_t_sem, edges_t, baseline=sr_t_mean - sr_t_sem,
                       color=SIM_COLOR, alpha=0.2, fill=True)
        for k, d in enumerate(ml_data):
            ax_main.stairs(ml_rtp_means[k], edges_t, color=d["color"], lw=1.5,
                           label=f"{d['label']} ({n})")
            ax_main.stairs(ml_rtp_means[k] + ml_rtp_sems[k], edges_t,
                           baseline=ml_rtp_means[k] - ml_rtp_sems[k],
                           color=d["color"], alpha=0.2, fill=True)
        ax_main.set_ylabel(r"Mean $t$ [$\mu$s]", fontsize=8)
        ax_main.set_xlim(0, R_MAX_M)
        ax_main.grid(False);  ax_main.legend(fontsize=7)
        ax_main.set_title("Radial Time Profile", fontsize=9)
        ax_ratio.set_ylim(0.5, 1.5);  ax_ratio.set_yticks([0.5, 1.0, 1.5])
    else:
        fig.add_subplot(outer_gs[row_i, 4]).text(0.5, 0.5, "No time data",
                                                  ha="center", va="center")

# ===========================================================================
# SAVE
# ===========================================================================

plt.savefig(OUTPUT_PDF, bbox_inches="tight")
print(f"Saved → {OUTPUT_PDF}")