import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ------------------------------------------------------------------ config

simulated_file = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_for_training/h5_files_v3/combined_electrons_test.h5"


# (filename, legend label, color)
ML_FILES = [
    ("/n/home04/hhanif/AllShowers/results/20260722_094249_Electron-Allshower/samples01.h5", "Attention windows size: 4; stepsize: 32",  "#2ca02c", "s"),
    ("/n/home04/hhanif/AllShowers/results/20260722_205728_Electron-Allshower/samples01.h5", "Attention windows size: 8; stepsize: 32",  "#1f77b4", "o"),
    # ("samples01.h5", "DPM-Solver++ ; stepsize: 8 (t=53s)",  "#ff7f0e", "s"),
    # ("samples02.h5", "DPM-Solver++ ; stepsize: 16 (t=81s)", "#2ca02c", "^"),
    # ("samples03.h5", "midpoint ; stepsize: 8 (t=86s)",      "#d62728", "D"),
    # ("samples04.h5", "midpoint ; stepsize: 16 (t=130s)",     "#9467bd", "v"),
    # ("samples05.h5", "midpoint ; stepsize: 32 (t=250s)",     "#05c0d1", ">"),
    # ("samples06.h5", "DPM-Solver++ ; stepsize: 32 (t=138s)",     "#2600ff", "o"),

]

RATIO_MARKER_ALPHA = 0.5  # opacity of ratio-panel markers (lower = more transparent)

NUM_LAYERS = 24
US = 1e6           # seconds -> microseconds
THRESHOLD = 1e-4   # GeV (= 0.1 MeV), matching reference script
SIM_COLOR = "black"


# ------------------------------------------------------------------ loaders

def load_file(path):
    print(f"  Reading {path} ...")
    with h5py.File(path, "r") as f:
        pdg   = f["pdg"][:]
        raw   = f["showers"][:]
        shape = f["shape"][:]

    N, max_pts, ncols = int(shape[0]), int(shape[1]), int(shape[2])
    print(f"  {N} showers, {max_pts} max pts, {ncols} cols")

    lengths = np.array([len(r) // ncols for r in raw], dtype=np.int32)
    flat_all = np.concatenate([np.asarray(r, dtype=np.float32) for r in raw])
    hits_all = flat_all.reshape(-1, ncols)

    pts = np.zeros((N, max_pts, ncols), dtype=np.float32)
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    for i, (start, L) in enumerate(zip(offsets[:-1], lengths)):
        if L > 0:
            pts[i, :L] = hits_all[start:start + L]

    pts[..., 3] = np.where(pts[..., 3] >= THRESHOLD, pts[..., 3], 0.0)

    return pts, pdg, ncols


def compute_longitudinal(pts, num_layers=NUM_LAYERS):
    N = pts.shape[0]
    energy_per_layer = np.zeros((N, num_layers), dtype=np.float64)
    layer_idx  = np.clip((pts[..., 2] + 0.1).astype(np.int32), 0, num_layers - 1)
    energies   = pts[..., 3].astype(np.float64)
    shower_idx = np.arange(N).reshape(-1, 1).repeat(pts.shape[1], axis=1)
    np.add.at(energy_per_layer, (shower_idx, layer_idx), energies)
    return energy_per_layer


def _cog_distances(pts_sel):
    dists = []
    for i in range(pts_sel.shape[0]):
        e = pts_sel[i, :, 3].astype(np.float64)
        mask = e > 0
        if mask.sum() == 0:
            continue
        e_h = e[mask]
        x_h = pts_sel[i, :, 0][mask].astype(np.float64)
        y_h = pts_sel[i, :, 1][mask].astype(np.float64)
        e_sum = e_h.sum()
        xc = (x_h * e_h).sum() / e_sum
        yc = (y_h * e_h).sum() / e_sum
        dists.append(np.sqrt((x_h - xc) ** 2 + (y_h - yc) ** 2))
    return np.concatenate(dists) if dists else np.array([1.0])


def compute_radial_profile(pts, n_events, num_bins=35, r_max=400.0):
    from scipy.stats import binned_statistic

    all_dist, all_e = [], []
    for i in range(pts.shape[0]):
        e = pts[i, :, 3].astype(np.float64)
        x = pts[i, :, 0].astype(np.float64)
        y = pts[i, :, 1].astype(np.float64)

        mask = e > 0
        if mask.sum() == 0:
            continue

        e_hit, x_hit, y_hit = e[mask], x[mask], y[mask]
        e_sum = e_hit.sum()
        x_cog = (x_hit * e_hit).sum() / e_sum
        y_cog = (y_hit * e_hit).sum() / e_sum
        dist = np.sqrt((x_hit - x_cog) ** 2 + (y_hit - y_cog) ** 2)
        all_dist.append(dist)
        all_e.append(e_hit)

    all_dist = np.concatenate(all_dist)
    all_e    = np.concatenate(all_e)

    mean,  edges, _ = binned_statistic(all_dist, all_e, bins=num_bins, statistic="mean",  range=(0, r_max))
    std,   _,     _ = binned_statistic(all_dist, all_e, bins=num_bins, statistic="std",   range=(0, r_max))
    count, _,     _ = binned_statistic(all_dist, all_e, bins=num_bins, statistic="count", range=(0, r_max))

    mean_shower = mean * count / n_events
    sem_shower  = (std / np.sqrt(count.clip(min=1))) * (count / n_events)

    return edges, mean_shower, sem_shower


def compute_radial_time_profile(pts, ncols, num_bins=35, r_max=400.0):
    if ncols < 5:
        return None, None, None
    from scipy.stats import binned_statistic

    all_dist, all_t = [], []
    for i in range(pts.shape[0]):
        e = pts[i, :, 3].astype(np.float64)
        x = pts[i, :, 0].astype(np.float64)
        y = pts[i, :, 1].astype(np.float64)
        t = pts[i, :, 4].astype(np.float64)

        mask = e > 0
        if mask.sum() == 0:
            continue

        e_hit, x_hit, y_hit, t_hit = e[mask], x[mask], y[mask], t[mask]
        e_sum = e_hit.sum()
        x_cog = (x_hit * e_hit).sum() / e_sum
        y_cog = (y_hit * e_hit).sum() / e_sum
        dist = np.sqrt((x_hit - x_cog) ** 2 + (y_hit - y_cog) ** 2)
        all_dist.append(dist)
        all_t.append(t_hit)

    all_dist = np.concatenate(all_dist)
    all_t    = np.concatenate(all_t)

    mean,  edges, _ = binned_statistic(all_dist, all_t, bins=num_bins, statistic="mean",  range=(0, r_max))
    std,   _,     _ = binned_statistic(all_dist, all_t, bins=num_bins, statistic="std",   range=(0, r_max))
    count, _,     _ = binned_statistic(all_dist, all_t, bins=num_bins, statistic="count", range=(0, r_max))

    sem = std / np.sqrt(count.clip(min=1))
    return edges, mean, sem


def mean_hit_time_per_shower(pts):
    N = pts.shape[0]
    result = np.full(N, np.nan)
    for i in range(N):
        mask = pts[i, :, 3] > 0
        if mask.sum() > 0:
            result[i] = pts[i, :, 4][mask].mean()
    return result[~np.isnan(result)] * US


# ------------------------------------------------------------------ ratio-panel helpers

def make_ratio_axes(fig, gs_cell, xlabel="", xscale="linear"):
    """Create the (main, ratio) axis pair inside one GridSpec cell."""
    inner = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs_cell, height_ratios=[3, 1], hspace=0.08
    )
    ax_main  = fig.add_subplot(inner[0])
    ax_ratio = fig.add_subplot(inner[1], sharex=ax_main)
    plt.setp(ax_main.get_xticklabels(), visible=False)

    ax_ratio.axhline(1.0, color="gray", lw=0.8, ls="--")
    ax_ratio.set_ylim(0.5, 1.5)
    ax_ratio.set_yticks([0.5, 1.0, 1.5])
    ax_ratio.set_ylabel("ML / Sim", fontsize=7)
    ax_ratio.tick_params(labelsize=7)
    ax_ratio.grid(False)
    ax_ratio.set_xlabel(xlabel, fontsize=8)
    if xscale == "log":
        ax_ratio.set_xscale("log")

    return ax_main, ax_ratio


def plot_ratio_markers(ax_ratio, x, sim_vals, ml_vals, sim_err, ml_err, color, marker="o"):
    """Ratio plotted as markers with error bars (no connecting line)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(sim_vals > 0, ml_vals / sim_vals, np.nan)
        ratio_err = ratio * np.sqrt(
            np.where(sim_vals > 0, (sim_err / sim_vals) ** 2, 0) +
            np.where(ml_vals  > 0, (ml_err  / ml_vals ) ** 2, 0)
        )
    ax_ratio.errorbar(
        x, ratio, yerr=ratio_err, fmt=marker, markersize=4,
        color=color, alpha=RATIO_MARKER_ALPHA, capsize=1.5, elinewidth=0.7, mew=0
    )


def plot_ratio_markers_counts(ax_ratio, x, sim_counts, ml_counts, color, marker="o"):
    """Ratio for histogram-based (count) panels."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(sim_counts > 0, ml_counts / sim_counts, np.nan)
        ratio_err = np.where(
            sim_counts > 0,
            ratio * np.sqrt(1.0 / np.where(ml_counts > 0, ml_counts, 1)
                          + 1.0 / np.where(sim_counts > 0, sim_counts, 1)),
            np.nan
        )
    ax_ratio.errorbar(
        x, ratio, yerr=ratio_err, fmt=marker, markersize=4,
        color=color, alpha=RATIO_MARKER_ALPHA, capsize=1.5, elinewidth=0.7, mew=0
    )


# ------------------------------------------------------------------ load data

print("Loading Simulated...")
s_pts, s_pdg, s_ncols = load_file(simulated_file)
s_long = compute_longitudinal(s_pts)

ml_data = []
for fpath, label, color, marker in ML_FILES:
    print(f"Loading ML: {label} ...")
    pts, pdg, ncols = load_file(fpath)
    ml_data.append({
        "label": label,
        "color": color,
        "marker": marker,
        "pts": pts,
        "pdg": pdg,
        "ncols": ncols,
        "long": compute_longitudinal(pts),
    })

print("Done.\n")

layers = np.arange(1, NUM_LAYERS + 1)

# equalize sample counts across sim + all ML files (random subsample, fixed seed)
n_common = min([s_pts.shape[0]] + [d["pts"].shape[0] for d in ml_data])
rng = np.random.default_rng(42)
s_idx = rng.choice(s_pts.shape[0], size=n_common, replace=False)
for d in ml_data:
    d["idx"] = rng.choice(d["pts"].shape[0], size=n_common, replace=False)

n = n_common
header = f"All  —  samples: {n}"

s_sel_pts = s_pts[s_idx]

# ------------------------------------------------------------------ figure layout (single "All" row)

NCOLS = 5
fig = plt.figure(figsize=(NCOLS * 4.2, 4.8))
outer_gs = gridspec.GridSpec(
    1, NCOLS, figure=fig,
    hspace=0.55, wspace=0.38,
    top=0.88, bottom=0.10, left=0.05, right=0.99,
)

# ---- Col 0: Longitudinal Energy Profile ----
sl = s_long[s_idx]
sl_mean = sl.mean(0)
sl_sem  = sl.std(0) / np.sqrt(len(s_idx))

ax_main, ax_ratio = make_ratio_axes(fig, outer_gs[0, 0], xlabel="Observing Plane")
ax_main.set_ylim(auto=True)
ax_ratio.set_ylim(0.8, 1.2)
ax_ratio.set_yticks([0.8, 0.9, 1.0, 1.1, 1.2])

ax_main.plot(layers, sl_mean, color=SIM_COLOR, lw=1.5, drawstyle="steps-mid", label="CORSIKA")
ax_main.fill_between(layers, sl_mean - sl_sem, sl_mean + sl_sem, alpha=0.15, color=SIM_COLOR, step="mid")

for d in ml_data:
    ml_l = d["long"][d["idx"]]
    ml_mean = ml_l.mean(0)
    ml_sem  = ml_l.std(0) / np.sqrt(len(d["idx"]))
    ax_main.plot(layers, ml_mean, color=d["color"], lw=1.3, ls="--", drawstyle="steps-mid", label=d["label"])
    ax_main.fill_between(layers, ml_mean - ml_sem, ml_mean + ml_sem, alpha=0.12, color=d["color"], step="mid")
    plot_ratio_markers(ax_ratio, layers, sl_mean, ml_mean, sl_sem, ml_sem, d["color"], d["marker"])

ax_main.set_ylabel("Mean Energy [GeV]", fontsize=8)
ax_main.set_xticks(list(np.arange(1, NUM_LAYERS + 1, 4)) + [NUM_LAYERS])
ax_main.grid(False)
ax_main.legend(fontsize=6, loc="best")
ax_main.set_title("Longitudinal Energy Profile", fontsize=9)
ax_main.annotate(header, xy=(0, 1.30), xycoords="axes fraction",
                  fontsize=10, fontweight="bold", color="#111111")

# ---- Col 1: Cell Energy Spectrum ----
s_ce = s_sel_pts[..., 3].ravel()
s_ce = s_ce[s_ce > 0]

ml_ce_list = []
for d in ml_data:
    m_ce = d["pts"][d["idx"]][..., 3].ravel()
    ml_ce_list.append(m_ce[m_ce > 0])

e_min = 1e-2
e_max = max([s_ce.max()] + [c.max() for c in ml_ce_list])
bins = np.logspace(np.log10(e_min), np.log10(e_max), 80)
bin_centers = np.sqrt(bins[:-1] * bins[1:])

s_counts, _ = np.histogram(s_ce, bins=bins)
s_err = np.sqrt(s_counts)

ax_main, ax_ratio = make_ratio_axes(fig, outer_gs[0, 1], xlabel="Cell energy [GeV]", xscale="log")
ax_ratio.set_ylim(0.5, 1.5)
ax_ratio.set_yticks([0.5, 1.0, 1.5])

ax_main.stairs(s_counts, bins, color=SIM_COLOR, lw=1.5, label="Simulated")
ax_main.stairs(s_counts + s_err, bins, baseline=s_counts - s_err, color=SIM_COLOR, alpha=0.2, fill=True)

for d, m_ce in zip(ml_data, ml_ce_list):
    m_counts, _ = np.histogram(m_ce, bins=bins)
    m_err = np.sqrt(m_counts)
    ax_main.stairs(m_counts, bins, color=d["color"], lw=1.3, ls="--", label=d["label"])
    ax_main.stairs(m_counts + m_err, bins, baseline=m_counts - m_err, color=d["color"], alpha=0.15, fill=True)
    plot_ratio_markers_counts(ax_ratio, bin_centers, s_counts, m_counts, d["color"], d["marker"])

ax_main.set_yscale("log")
ax_main.set_xscale("log")
ax_main.set_xlim(e_min, e_max)
ax_main.set_ylabel("Number of cells", fontsize=8)
ax_main.grid(False)
ax_main.legend(fontsize=6)
ax_main.set_title("Cell Energy Spectrum", fontsize=9)
for ax in [ax_main, ax_ratio]:
    ax.tick_params(labelsize=7)

# ---- Col 2: Radial Energy Profile ----
all_r = np.concatenate([_cog_distances(s_sel_pts)] + [_cog_distances(d["pts"][d["idx"]]) for d in ml_data])
R_MAX_M = float(np.percentile(all_r, 99))

edges, sr_mean, sr_sem = compute_radial_profile(s_sel_pts, n_events=n, r_max=R_MAX_M)
sr_mean *= 1e3; sr_sem *= 1e3
bin_centers = 0.5 * (edges[:-1] + edges[1:])

ax_main, ax_ratio = make_ratio_axes(fig, outer_gs[0, 2], xlabel="Radial Distance [m]")
ax_ratio.set_ylim(0.5, 1.5)
ax_ratio.set_yticks([0.5, 1.0, 1.5])

ax_main.stairs(sr_mean, edges, color=SIM_COLOR, lw=1.5, label="Simulated")
ax_main.stairs(sr_mean + sr_sem, edges, baseline=sr_mean - sr_sem, color=SIM_COLOR, alpha=0.2, fill=True)

for d in ml_data:
    _, mr_mean, mr_sem = compute_radial_profile(d["pts"][d["idx"]], n_events=n, r_max=R_MAX_M)
    mr_mean *= 1e3; mr_sem *= 1e3
    ax_main.stairs(mr_mean, edges, color=d["color"], lw=1.3, ls="--", label=d["label"])
    ax_main.stairs(mr_mean + mr_sem, edges, baseline=mr_mean - mr_sem, color=d["color"], alpha=0.15, fill=True)
    plot_ratio_markers(ax_ratio, bin_centers, sr_mean, mr_mean, sr_sem, mr_sem, d["color"], d["marker"])

ax_main.set_yscale("log")
ax_main.set_ylabel("Mean Energy [MeV]", fontsize=8)
ax_main.set_xlim(0, R_MAX_M)
ax_main.grid(False)
ax_main.legend(fontsize=6)
ax_main.set_title("Radial Energy Profile", fontsize=9)

# ---- Col 3: Mean Hit Time per Shower ----
if s_ncols >= 5:
    s_mht = mean_hit_time_per_shower(s_sel_pts)
    ml_mht_list = [mean_hit_time_per_shower(d["pts"][d["idx"]]) for d in ml_data if d["ncols"] >= 5]

    t_lo = min([s_mht.min()] + [m.min() for m in ml_mht_list])
    t_hi = max([s_mht.max()] + [m.max() for m in ml_mht_list])
    t_bins = np.linspace(t_lo, t_hi, 40)
    t_centers = 0.5 * (t_bins[:-1] + t_bins[1:])

    s_tcounts, _ = np.histogram(s_mht, bins=t_bins)
    s_terr = np.sqrt(s_tcounts)

    ax_main, ax_ratio = make_ratio_axes(fig, outer_gs[0, 3], xlabel=r"Mean $t$ [$\mu$s]")
    ax_ratio.set_ylim(0.5, 1.5)
    ax_ratio.set_yticks([0.5, 1.0, 1.5])

    ax_main.stairs(s_tcounts, t_bins, color=SIM_COLOR, lw=1.5, label="Simulated")
    ax_main.stairs(s_tcounts + s_terr, t_bins, baseline=s_tcounts - s_terr, color=SIM_COLOR, alpha=0.2, fill=True)

    for d in ml_data:
        if d["ncols"] < 5:
            continue
        m_mht = mean_hit_time_per_shower(d["pts"][d["idx"]])
        m_tcounts, _ = np.histogram(m_mht, bins=t_bins)
        ax_main.stairs(m_tcounts, t_bins, color=d["color"], lw=1.3, ls="--", label=d["label"])
        m_terr = np.sqrt(m_tcounts)
        ax_main.stairs(m_tcounts + m_terr, t_bins, baseline=m_tcounts - m_terr, color=d["color"], alpha=0.15, fill=True)
        plot_ratio_markers_counts(ax_ratio, t_centers, s_tcounts, m_tcounts, d["color"], d["marker"])

    ax_main.set_ylabel("Number of Showers", fontsize=8)
    ax_main.set_title("Mean Hit Time per Shower", fontsize=9)
    ax_main.legend(fontsize=6)
    ax_main.grid(False)
    for ax in [ax_main, ax_ratio]:
        ax.tick_params(labelsize=7)
else:
    fig.add_subplot(outer_gs[0, 3]).text(0.5, 0.5, "No time data", ha="center", va="center")

# ---- Col 4: Radial Time Profile ----
edges_t, sr_t_mean, sr_t_sem = compute_radial_time_profile(s_sel_pts, s_ncols, r_max=R_MAX_M)

if edges_t is not None:
    sr_t_mean *= US; sr_t_sem *= US
    t_bin_centers = 0.5 * (edges_t[:-1] + edges_t[1:])

    ax_main, ax_ratio = make_ratio_axes(fig, outer_gs[0, 4], xlabel="Radial Distance [m]")
    ax_ratio.set_ylim(0.5, 1.5)
    ax_ratio.set_yticks([0.5, 1.0, 1.5])

    ax_main.stairs(sr_t_mean, edges_t, color=SIM_COLOR, lw=1.5, label="Simulated")
    ax_main.stairs(sr_t_mean + sr_t_sem, edges_t, baseline=sr_t_mean - sr_t_sem, color=SIM_COLOR, alpha=0.2, fill=True)

    for d in ml_data:
        _, mr_t_mean, mr_t_sem = compute_radial_time_profile(d["pts"][d["idx"]], d["ncols"], r_max=R_MAX_M)
        if mr_t_mean is None:
            continue
        mr_t_mean *= US; mr_t_sem *= US
        ax_main.stairs(mr_t_mean, edges_t, color=d["color"], lw=1.3, ls="--", label=d["label"])
        ax_main.stairs(mr_t_mean + mr_t_sem, edges_t, baseline=mr_t_mean - mr_t_sem, color=d["color"], alpha=0.15, fill=True)
        plot_ratio_markers(ax_ratio, t_bin_centers, sr_t_mean, mr_t_mean, sr_t_sem, mr_t_sem, d["color"], d["marker"])

    ax_main.set_ylabel(r"Mean $t$ [$\mu$s]", fontsize=8)
    ax_main.set_xlim(0, R_MAX_M)
    ax_main.grid(False)
    ax_main.legend(fontsize=6)
    ax_main.set_title("Radial Time Profile", fontsize=9)
else:
    fig.add_subplot(outer_gs[0, 4]).text(0.5, 0.5, "No time data", ha="center", va="center")

out = "/n/home04/hhanif/tambo_plots/plot_attention_windows_electrons_stepsize32.pdf"
plt.savefig(out, bbox_inches="tight")
print(f"Saved → {out}")