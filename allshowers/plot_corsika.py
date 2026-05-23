import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ml_file        = "/n/home04/hhanif/AllShowers/results/20260519_185649_Electron-Allshower/samples00.h5"
# simulated_file = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_for_training/h5_files_v3/combined_electrons_test.h5"

ml_file        = "/n/home04/hhanif/AllShowers/results/20260521_074401_Photon-Allshower/samples01.h5"
simulated_file = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_for_training/h5_files_v3/combined_photons_test.h5"

# ml_file        = "/n/home04/hhanif/AllShowers/results/20260520_160031_Muons-Allshower/samples00.h5"
# simulated_file = "/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations_for_training/h5_files_v3/combined_muons_test.h5"


CLASS_NAMES = {
    0: r"$e^\pm$/$\gamma$/$\pi^0$",
    1: r"$\pi^\pm$",
}
NUM_LAYERS = 24
US = 1e6  # seconds -> microseconds
THRESHOLD = 1e-4  # GeV (= 0.1 MeV), matching reference script


# ------------------------------------------------------------------ loaders

def load_file(path):
    print(f"  Reading {path} ...")
    with h5py.File(path, "r") as f:
        pdg   = f["pdg"][:]
        raw   = f["showers"][:]
        shape = f["shape"][:]

    N, max_pts, ncols = int(shape[0]), int(shape[1]), int(shape[2])
    print(f"  {N} showers, {max_pts} max pts, {ncols} cols")

    pts = np.zeros((N, max_pts, ncols), dtype=np.float32)
    for i, flat in enumerate(raw):
        arr = np.asarray(flat, dtype=np.float32).reshape(-1, ncols)
        pts[i, :len(arr)] = arr

    # zero out sub-threshold hits
    pts[..., 3] = np.where(pts[..., 3] >= THRESHOLD, pts[..., 3], 0.0)

    return pts, pdg, ncols, raw


def compute_longitudinal(pts, num_layers=NUM_LAYERS):
    """Mean energy per layer across showers (not normalised)."""
    N = pts.shape[0]
    energy_per_layer = np.zeros((N, num_layers), dtype=np.float64)
    layer_idx  = np.clip((pts[..., 2] + 0.1).astype(np.int32), 0, num_layers - 1)
    energies   = pts[..., 3].astype(np.float64)
    shower_idx = np.arange(N).reshape(-1, 1).repeat(pts.shape[1], axis=1)
    np.add.at(energy_per_layer, (shower_idx, layer_idx), energies)
    return energy_per_layer  # shape (N, num_layers)


def compute_time_per_layer(pts, ncols, num_layers=NUM_LAYERS):
    """Mean hit time per layer per shower."""
    if ncols < 5:
        return None
    N = pts.shape[0]
    time_sum   = np.zeros((N, num_layers), dtype=np.float64)
    time_count = np.zeros((N, num_layers), dtype=np.float64)
    mask       = pts[..., 3] > 0
    layer_idx  = np.clip((pts[..., 2] + 0.1).astype(np.int32), 0, num_layers - 1)
    t          = pts[..., 4].astype(np.float64)

    for i in range(N):
        m = mask[i]
        li = layer_idx[i][m]
        ti = t[i][m]
        np.add.at(time_sum[i],   li, ti)
        np.add.at(time_count[i], li, 1)

    return time_sum / time_count.clip(min=1)  # shape (N, num_layers)


def compute_cell_energies(pts):
    """Flat array of all nonzero hit energies across all showers."""
    e = pts[..., 3].ravel()
    return e[e > 0]


def compute_longitudinal_time(pts, ncols, num_layers=NUM_LAYERS):
    """Mean hit time per layer, averaged across showers. shape (N, num_layers)."""
    if ncols < 5:
        return None
    return compute_time_per_layer(pts, ncols, num_layers)


def compute_cell_times(pts, ncols):
    """Flat array of hit times for all hits with energy > threshold."""
    if ncols < 5:
        return None
    mask = pts[..., 3] > 0
    return pts[..., 4][mask].astype(np.float64)


def compute_radial_time_profile(pts, ncols, n_events, num_bins=35, r_max=400.0):
    """
    Mean hit time per radial bin (CoG-centred), scaled by count/n_events.
    Same structure as compute_radial_profile but for the time column.
    """
    if ncols < 5:
        return None, None, None
    from scipy.stats import binned_statistic

    all_dist = []
    all_t    = []

    for i in range(pts.shape[0]):
        e = pts[i, :, 3].astype(np.float64)
        x = pts[i, :, 0].astype(np.float64)
        y = pts[i, :, 1].astype(np.float64)
        t = pts[i, :, 4].astype(np.float64)

        mask = e > 0
        if mask.sum() == 0:
            continue

        e_hit = e[mask];  x_hit = x[mask];  y_hit = y[mask];  t_hit = t[mask]
        e_sum = e_hit.sum()
        x_cog = (x_hit * e_hit).sum() / e_sum
        y_cog = (y_hit * e_hit).sum() / e_sum
        dist  = np.sqrt((x_hit - x_cog)**2 + (y_hit - y_cog)**2)
        all_dist.append(dist)
        all_t.append(t_hit)

    all_dist = np.concatenate(all_dist)
    all_t    = np.concatenate(all_t)

    mean,  edges, _ = binned_statistic(all_dist, all_t, bins=num_bins, statistic="mean",  range=(0, r_max))
    std,   _,     _ = binned_statistic(all_dist, all_t, bins=num_bins, statistic="std",   range=(0, r_max))
    count, _,     _ = binned_statistic(all_dist, all_t, bins=num_bins, statistic="count", range=(0, r_max))

    mean_shower = mean * count / n_events
    sem_shower  = (std / np.sqrt(count.clip(min=1))) * (count / n_events)

    return edges, mean_shower, sem_shower


def compute_radial_profile(pts, n_events, num_bins=35, r_max=400.0):
    """
    Compute mean shower energy per radial bin, matching the reference script.
    Hits are centered per-shower using the energy-weighted centre of gravity
    before computing radial distance, since x/y are absolute detector coords.
    """
    from scipy.stats import binned_statistic

    all_dist = []
    all_e    = []

    for i in range(pts.shape[0]):
        e = pts[i, :, 3].astype(np.float64)
        x = pts[i, :, 0].astype(np.float64)
        y = pts[i, :, 1].astype(np.float64)

        mask = e > 0
        if mask.sum() == 0:
            continue

        e_hit = e[mask]
        x_hit = x[mask]
        y_hit = y[mask]

        # energy-weighted centre
        e_sum = e_hit.sum()
        x_cog = (x_hit * e_hit).sum() / e_sum
        y_cog = (y_hit * e_hit).sum() / e_sum

        dist = np.sqrt((x_hit - x_cog)**2 + (y_hit - y_cog)**2)
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


# ------------------------------------------------------------------ load

print("Loading Simulated...")
s_pts, s_pdg, s_ncols, s_raw = load_file(simulated_file)

print("Loading ML...")
m_pts, m_pdg, m_ncols, m_raw = load_file(ml_file)

print("Computing observables...")
s_long   = compute_longitudinal(s_pts)
m_long   = compute_longitudinal(m_pts)
s_tplane = compute_time_per_layer(s_pts, s_ncols)
m_tplane = compute_time_per_layer(m_pts, m_ncols)
# radial profiles computed per-row using capped indices (done inside the plot loop)
radial_bin_edges = np.linspace(0, 400.0, 36)  # 35 bins, stored for axis use
radial_bin_centers = 0.5 * (radial_bin_edges[:-1] + radial_bin_edges[1:])

print("Done.\n")

layers = np.arange(1, NUM_LAYERS + 1)

# ------------------------------------------------------------------ helpers

def mask_for(pdg_arr, pdg_val):
    return np.ones(len(pdg_arr), dtype=bool) if pdg_val is None else pdg_arr == pdg_val


def capped_indices(s_mask, m_mask, seed=42):
    s_idx = np.where(s_mask)[0]
    m_idx = np.where(m_mask)[0]
    n = min(len(s_idx), len(m_idx))
    rng = np.random.default_rng(seed)
    s_idx = rng.choice(s_idx, size=n, replace=False)
    m_idx = rng.choice(m_idx, size=n, replace=False)
    return s_idx, m_idx, n


def add_ratio_panel(fig, gs_cell, x, sim_vals, ml_vals, sim_err=None, ml_err=None,
                    xscale="linear", xlabel=""):
    """
    Creates a (main, ratio) pair of axes inside a single GridSpec cell.
    Returns (ax_main, ax_ratio).
    """
    inner = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs_cell,
        height_ratios=[3, 1], hspace=0.08
    )
    ax_main  = fig.add_subplot(inner[0])
    ax_ratio = fig.add_subplot(inner[1], sharex=ax_main)
    plt.setp(ax_main.get_xticklabels(), visible=False)

    # ratio  ML / Sim
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(sim_vals > 0, ml_vals / sim_vals, np.nan)

    ax_ratio.axhline(1.0, color="gray", lw=0.8, ls="--")
    ax_ratio.plot(x, ratio, color="steelblue", lw=1.2, drawstyle="steps-mid" if xscale == "linear" else "default")

    if sim_err is not None and ml_err is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_err = ratio * np.sqrt(
                np.where(sim_vals > 0, (sim_err / sim_vals)**2, 0) +
                np.where(ml_vals  > 0, (ml_err  / ml_vals )**2, 0)
            )
        ax_ratio.fill_between(x, ratio - ratio_err, ratio + ratio_err, alpha=0.25, color="steelblue", step="mid" if xscale=="linear" else None)

    ax_ratio.set_ylim(0.5, 1.5)
    ax_ratio.set_yticks([0.95, 1.0, 1.05])
    ax_ratio.set_ylabel("ML / Sim", fontsize=7)
    ax_ratio.tick_params(labelsize=7)
    ax_ratio.grid(False)
    ax_ratio.set_xlabel(xlabel, fontsize=8)
    if xscale == "log":
        ax_ratio.set_xscale("log")

    return ax_main, ax_ratio


# ------------------------------------------------------------------ figure layout

row_configs = [
    ("All", None),
    (f"Class 0: {CLASS_NAMES[0]}", 0),
    (f"Class 1: {CLASS_NAMES[1]}", 1),
]

NROWS = len(row_configs)
NCOLS = 5

fig = plt.figure(figsize=(NCOLS * 4.2, NROWS * 4.8))
outer_gs = gridspec.GridSpec(
    NROWS, NCOLS,
    figure=fig,
    hspace=0.55, wspace=0.38,
    top=0.96, bottom=0.05, left=0.05, right=0.99,
)

SIM_COLOR = "black"
ML_COLOR  = "steelblue"

for row_i, (row_label, class_val) in enumerate(row_configs):
    s_mask = mask_for(s_pdg, class_val)
    m_mask = mask_for(m_pdg, class_val)
    s_idx, m_idx, n = capped_indices(s_mask, m_mask)

    header = f"{'All' if class_val is None else row_label}  —  samples: {n}"

    # ---- Col 0: Longitudinal Energy Profile ----
    sl = s_long[s_idx]
    ml_l = m_long[m_idx]
    sl_mean = sl.mean(0)
    sl_sem  = sl.std(0) / np.sqrt(len(s_idx))
    ml_mean = ml_l.mean(0)
    ml_sem  = ml_l.std(0) / np.sqrt(len(m_idx))

    ax_main, ax_ratio = add_ratio_panel(
        fig, outer_gs[row_i, 0],
        layers, sl_mean, ml_mean, sl_sem, ml_sem,
        xlabel="Plane"
    )
    ax_main.plot(layers, sl_mean, color=SIM_COLOR, lw=1.5, drawstyle="steps-mid", label=f"Simulated ({n})")
    ax_main.fill_between(layers, sl_mean - sl_sem, sl_mean + sl_sem, alpha=0.15, color=SIM_COLOR, step="mid")
    ax_main.plot(layers, ml_mean, color=ML_COLOR,  lw=1.5, drawstyle="steps-mid", label=f"ML ({n})")
    ax_main.fill_between(layers, ml_mean - ml_sem, ml_mean + ml_sem, alpha=0.20, color=ML_COLOR,  step="mid")
    ax_main.set_ylabel("Mean Energy [GeV]", fontsize=8)
    ax_main.set_xticks(np.arange(1, NUM_LAYERS + 1, 4))
    ax_main.grid(False)
    ax_main.legend(fontsize=7)
    ax_main.set_title("Longitudinal Energy Profile", fontsize=9)
    ax_main.annotate(
        header,
        xy=(0, 1.22), xycoords="axes fraction",
        fontsize=10, fontweight="bold", color="#111111",
    )

    # ---- Col 1: Cell Energy Spectrum ----
    s_ce = s_pts[s_idx][..., 3].ravel()
    m_ce = m_pts[m_idx][..., 3].ravel()
    s_ce = s_ce[s_ce > 0]
    m_ce = m_ce[m_ce > 0]

    e_min = 1e-2  # start at 10^-2 GeV
    e_max = max(s_ce.max(), m_ce.max())
    bins  = np.logspace(np.log10(e_min), np.log10(e_max), 80)

    s_counts, _ = np.histogram(s_ce, bins=bins)
    m_counts, _ = np.histogram(m_ce, bins=bins)
    bin_centers  = np.sqrt(bins[:-1] * bins[1:])
    s_err = np.sqrt(s_counts)
    m_err = np.sqrt(m_counts)

    inner = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer_gs[row_i, 1],
        height_ratios=[3, 1], hspace=0.08
    )
    ax_main  = fig.add_subplot(inner[0])
    ax_ratio = fig.add_subplot(inner[1], sharex=ax_main)
    plt.setp(ax_main.get_xticklabels(), visible=False)

    ax_main.stairs(s_counts, bins, color=SIM_COLOR, lw=1.5, label=f"Simulated ({n})")
    ax_main.errorbar(bin_centers, s_counts, yerr=s_err, fmt='none', color=SIM_COLOR, lw=0.8, capsize=2)
    ax_main.stairs(m_counts, bins, color=ML_COLOR,  lw=1.5, label=f"ML ({n})")
    ax_main.errorbar(bin_centers, m_counts, yerr=m_err, fmt='none', color=ML_COLOR,  lw=0.8, capsize=2)
    ax_main.set_yscale("log")
    ax_main.set_xscale("log")
    ax_main.set_xlim(e_min, e_max)
    ax_main.set_ylabel("Number of cells", fontsize=8)
    ax_main.grid(False)
    ax_main.legend(fontsize=7)
    ax_main.set_title("Cell Energy Spectrum", fontsize=9)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(s_counts > 0, m_counts / s_counts, np.nan)
        ratio_err = np.where(s_counts > 0,
            ratio * np.sqrt(1.0 / np.where(m_counts > 0, m_counts, 1)
                          + 1.0 / np.where(s_counts > 0, s_counts, 1)),
            np.nan)
    ax_ratio.axhline(1.0, color="gray", lw=0.8, ls="--")
    ax_ratio.plot(bin_centers, ratio, color=ML_COLOR, lw=1.2)
    ax_ratio.fill_between(bin_centers, ratio - ratio_err, ratio + ratio_err, alpha=0.25, color=ML_COLOR)
    ax_ratio.set_xscale("log")
    ax_ratio.set_ylim(0.5, 1.5)
    ax_ratio.set_yticks([0.5, 1.0, 1.5])
    ax_ratio.set_ylabel("ML / Sim", fontsize=7)
    ax_ratio.set_xlabel("Cell energy [GeV]", fontsize=8)
    ax_ratio.tick_params(labelsize=7)
    ax_ratio.grid(False)

    for ax in [ax_main, ax_ratio]:
        ax.tick_params(labelsize=7)

    # ---- Col 2: Radial Energy Profile ----
    # x/y are already in meters; compute r_max from 99th percentile of CoG-centred distances
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
            dists.append(np.sqrt((x_h - xc)**2 + (y_h - yc)**2))
        return np.concatenate(dists) if dists else np.array([1.0])

    all_r = np.concatenate([_cog_distances(s_pts[s_idx]), _cog_distances(m_pts[m_idx])])
    R_MAX_M = float(np.percentile(all_r, 99))

    edges, sr_mean, sr_sem = compute_radial_profile(s_pts[s_idx], n_events=len(s_idx), r_max=R_MAX_M)
    _,     mr_mean, mr_sem = compute_radial_profile(m_pts[m_idx], n_events=len(m_idx), r_max=R_MAX_M)

    # convert GeV -> MeV
    sr_mean *= 1e3;  sr_sem *= 1e3
    mr_mean *= 1e3;  mr_sem *= 1e3
    bin_centers = 0.5 * (edges[:-1] + edges[1:])

    ax_main, ax_ratio = add_ratio_panel(
        fig, outer_gs[row_i, 2],
        bin_centers, sr_mean, mr_mean, sr_sem, mr_sem,
        xlabel="Radial Distance [m]"
    )
    ax_main.stairs(sr_mean, edges, color=SIM_COLOR, lw=1.5, label=f"Simulated ({n})")
    ax_main.stairs(sr_mean + sr_sem, edges, baseline=sr_mean - sr_sem,
                   color=SIM_COLOR, alpha=0.2, fill=True)
    ax_main.stairs(mr_mean, edges, color=ML_COLOR, lw=1.5, label=f"ML ({n})")
    ax_main.stairs(mr_mean + mr_sem, edges, baseline=mr_mean - mr_sem,
                   color=ML_COLOR, alpha=0.2, fill=True)
    ax_main.set_yscale("log")
    ax_main.set_ylabel("Mean Energy [MeV]", fontsize=8)
    ax_main.set_xlim(0, R_MAX_M)
    ax_main.grid(False)
    ax_main.legend(fontsize=7)
    ax_main.set_title("Radial Energy Profile", fontsize=9)
    ax_ratio.set_ylim(0.5, 1.5)
    ax_ratio.set_yticks([0.5, 1.0, 1.5])

    # ---- Col 3: Longitudinal Time Profile ----
    if s_tplane is not None and m_tplane is not None:
        st = s_tplane[s_idx]
        mt = m_tplane[m_idx]
        st_mean = st.mean(0) * US
        st_sem  = st.std(0) / np.sqrt(len(s_idx)) * US
        mt_mean = mt.mean(0) * US
        mt_sem  = mt.std(0) / np.sqrt(len(m_idx)) * US

        ax_main, ax_ratio = add_ratio_panel(
            fig, outer_gs[row_i, 3],
            layers, st_mean, mt_mean, st_sem, mt_sem,
            xlabel="Plane"
        )
        ax_main.plot(layers, st_mean, color=SIM_COLOR, lw=1.5, drawstyle="steps-mid", label=f"Simulated ({n})")
        ax_main.fill_between(layers, st_mean - st_sem, st_mean + st_sem, alpha=0.15, color=SIM_COLOR, step="mid")
        ax_main.plot(layers, mt_mean, color=ML_COLOR,  lw=1.5, drawstyle="steps-mid", label=f"ML ({n})")
        ax_main.fill_between(layers, mt_mean - mt_sem, mt_mean + mt_sem, alpha=0.20, color=ML_COLOR,  step="mid")
        ax_main.set_ylabel(r"Mean $t$ [$\mu$s]", fontsize=8)
        ax_main.set_xticks(np.arange(1, NUM_LAYERS + 1, 4))
        ax_main.grid(False)
        ax_main.legend(fontsize=7)
        ax_main.set_title("Longitudinal Time Profile", fontsize=9)
        ax_ratio.set_ylim(0.5, 1.5)
        ax_ratio.set_yticks([0.5, 1.0, 1.5])
    else:
        fig.add_subplot(outer_gs[row_i, 3]).text(0.5, 0.5, "No time data", ha="center", va="center")

    # ---- Col 4: Radial Time Profile ----
    edges_t, sr_t_mean, sr_t_sem = compute_radial_time_profile(s_pts[s_idx], s_ncols, n_events=len(s_idx), r_max=R_MAX_M)
    _,        mr_t_mean, mr_t_sem = compute_radial_time_profile(m_pts[m_idx], m_ncols, n_events=len(m_idx), r_max=R_MAX_M)

    if edges_t is not None:
        sr_t_mean *= US;  sr_t_sem *= US
        mr_t_mean *= US;  mr_t_sem *= US
        t_bin_centers = 0.5 * (edges_t[:-1] + edges_t[1:])

        ax_main, ax_ratio = add_ratio_panel(
            fig, outer_gs[row_i, 4],
            t_bin_centers, sr_t_mean, mr_t_mean, sr_t_sem, mr_t_sem,
            xlabel="Radial Distance [m]"
        )
        ax_main.stairs(sr_t_mean, edges_t, color=SIM_COLOR, lw=1.5, label=f"Simulated ({n})")
        ax_main.stairs(sr_t_mean + sr_t_sem, edges_t, baseline=sr_t_mean - sr_t_sem,
                       color=SIM_COLOR, alpha=0.2, fill=True)
        ax_main.stairs(mr_t_mean, edges_t, color=ML_COLOR, lw=1.5, label=f"ML ({n})")
        ax_main.stairs(mr_t_mean + mr_t_sem, edges_t, baseline=mr_t_mean - mr_t_sem,
                       color=ML_COLOR, alpha=0.2, fill=True)
        ax_main.set_ylabel(r"Mean $t$ [$\mu$s]", fontsize=8)
        ax_main.set_xlim(0, R_MAX_M)
        ax_main.grid(False)
        ax_main.legend(fontsize=7)
        ax_main.set_title("Radial Time Profile", fontsize=9)
        ax_ratio.set_ylim(0.5, 1.5)
        ax_ratio.set_yticks([0.5, 1.0, 1.5])
    else:
        fig.add_subplot(outer_gs[row_i, 4]).text(0.5, 0.5, "No time data", ha="center", va="center")

out = "shower_observables_reference_style_photon_4.png"
plt.savefig(out, dpi=300, bbox_inches="tight")
print(f"Saved → {out}")