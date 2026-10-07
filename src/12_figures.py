"""
12 - Figures 3-8 of the manuscript (vector PDF and 600 dpi PNG at final print size).

Sizes follow the journal's artwork guidance: full text width 165 mm, fonts 7-8 pt at final size.
After drawing, every figure is checked for overlapping text and for text outside the canvas; problems are printed.
Usage: python 12_figures.py [fig3 fig4 ...]
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator

import config as C

MM = 1 / 25.4
FULL = 165 * MM
INK, INK2, GRID = "#1a1a1a", "#555555", "#e3e3e3"
COL = {"ARIMA": "#2a78d6", "RF": "#eb6834", "XGB": "#1baf7a", "LGBM": "#eda100", "LSTM": "#e87ba4",
       "Persistence": "#7a7974"}
MARK = {"ARIMA": "o", "RF": "s", "XGB": "^", "LGBM": "D", "LSTM": "v", "Persistence": "X"}
LABEL = {"ARIMA": "ARIMA", "RF": "RF", "XGB": "XGBoost", "LGBM": "LightGBM", "LSTM": "LSTM", "Persistence": "Persistence"}
ICOL = {"Linear": "#2a78d6", "Quadratic": "#eb6834", "Spline": "#1baf7a"}
IMARK = {"Linear": "o", "Quadratic": "s", "Spline": "^"}
ILABEL = {"Linear": "Linear", "Quadratic": "Quadratic", "Spline": "Cubic spline"}
INC, DEC, NS = "#2a78d6", "#d63c3b", "#c8c8c8"
MODELS = ["ARIMA", "RF", "XGB", "LGBM", "LSTM"]

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 7.5,
    "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False, "grid.color": GRID, "grid.linewidth": 0.5,
    "legend.frameon": False, "savefig.facecolor": "white", "pdf.fonttype": 42, "ps.fonttype": 42,
})


# ---------------------------------------------------------------- quality check
def check_layout(fig, name):
    """Reports overlapping text and text that extends beyond the canvas."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    hidden = set()   # tick labels outside the axis limits are not drawn
    for ax in fig.axes:
        for axis, lim in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            lo, hi = min(lim), max(lim)
            for tick in axis.get_major_ticks():
                if not lo - 1e-9 <= tick.get_loc() <= hi + 1e-9:
                    hidden.update({id(tick.label1), id(tick.label2)})
    texts = [t for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text().strip()
             and id(t) not in hidden and t.get_window_extent(r).width > 0]
    boxes = [(t, t.get_window_extent(r).padded(-0.5)) for t in texts]
    problems = []
    for t, b in boxes:
        if b.x0 < -1 or b.y0 < -1 or b.x1 > W + 1 or b.y1 > H + 1:
            problems.append(f"outside canvas: '{t.get_text()[:30]}'")
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i][1].overlaps(boxes[j][1]):
                problems.append(f"overlap: '{boxes[i][0].get_text()[:25]}' <-> '{boxes[j][0].get_text()[:25]}'")
    print(f"{name}: {len(texts)} text elements, {'no problems' if not problems else str(len(problems)) + ' problems'}")
    for p in problems:
        print("   ", p)
    return problems


def save(fig, name):
    check_layout(fig, name)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
        fig.savefig(os.path.join(C.FIGURES, f"{name}.{ext}"), **kw)
    plt.close(fig)


def title(ax, letter, text):
    ax.set_title(f"({letter}) {text}", loc="left", fontsize=8, color=INK, pad=4)


def boot_median(x, rng, B=2000):
    x = np.asarray(x)
    return np.percentile(np.median(x[rng.integers(0, len(x), (B, len(x)))], axis=1), [2.5, 97.5])


def forecast_table():
    m = pd.read_csv(C.out("forecast_metrics_long.csv"), dtype={"Station_id": str})
    w = m[m.error.isna()].pivot(index="Station_id", columns="model", values="RMSE")
    w = w.dropna(subset=["Persistence"] + MODELS)
    status = pd.read_csv(C.out("forecast_status.csv"), dtype={"Station_id": str}).set_index("Station_id")
    return w, status


# ---------------------------------------------------------------- figures
def fig3():
    rng = np.random.default_rng(1)
    d = pd.read_csv(C.out("interpolation_runs.csv"), dtype={"Station_id": str})
    st = d.groupby(["scheme", "rate_pct", "Station_id", "method"]).RMSE.mean().reset_index()
    gl = pd.read_csv(C.out("gap_length_runs.csv"), dtype={"Station_id": str})
    gs = gl.groupby(["gap_len", "Station_id", "method"]).RMSE.mean().reset_index()
    fig, axs = plt.subplots(1, 3, figsize=(FULL, 62 * MM), layout="constrained")
    for ax, scheme, text, letter in [(axs[0], "random", "Random single days", "a"),
                                     (axs[1], "block", "Real gap lengths", "b")]:
        for meth in ICOL:
            g = st[(st.scheme == scheme) & (st.method == meth)]
            rates = sorted(g.rate_pct.unique())
            med = [g[g.rate_pct == r].RMSE.median() for r in rates]
            ci = np.array([boot_median(g[g.rate_pct == r].RMSE.values, rng) for r in rates])
            ax.fill_between(rates, ci[:, 0], ci[:, 1], color=ICOL[meth], alpha=0.15, lw=0)
            ax.plot(rates, med, color=ICOL[meth], marker=IMARK[meth], ms=3.5, lw=1.3, label=ILABEL[meth])
        ax.set_xticks([5, 10, 15, 20])
        ax.set_xlabel("Proportion of removed days (%)")
        ax.grid(axis="y")
        title(ax, letter, text)
    axs[0].set_ylabel("Median station RMSE (m)")
    ax = axs[2]
    for meth in ICOL:
        g = gs[gs.method == meth].groupby("gap_len").RMSE.median()
        ax.plot(g.index, g.values, color=ICOL[meth], marker=IMARK[meth], ms=3.2, lw=1.3)
    ax.axvline(14, color=INK2, lw=0.7, ls="--")
    ax.text(13, 0.43, "14-day\nfilling limit", fontsize=6.5, color=INK2, ha="right", va="top")
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 30, 90]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(["1", "2", "5", "10", "30", "90"])
    ax.set_xlabel("Gap length (days)")
    ax.set_ylim(0, 0.48)
    ax.grid(axis="y")
    title(ax, "c", "Single gaps of fixed length")
    handles = [Line2D([], [], color=ICOL[m], marker=IMARK[m], ms=3.5, lw=1.3, label=ILABEL[m]) for m in ICOL]
    fig.legend(handles=handles, loc="outside lower center", ncol=3)
    save(fig, "Fig3_interpolation")


def fig4():
    w, _ = forecast_table()
    skill = 1 - w.div(w["Persistence"], axis=0)
    fig, axs = plt.subplots(1, 2, figsize=(FULL, 72 * MM), layout="constrained", gridspec_kw={"width_ratios": [1, 1.05]})
    ax = axs[0]
    bp = ax.boxplot([skill[m].values for m in MODELS], widths=0.55, showfliers=False, patch_artist=True,
                    medianprops=dict(color=INK, lw=1.1), whiskerprops=dict(color=INK2, lw=0.6),
                    capprops=dict(color=INK2, lw=0.6))
    for patch, m in zip(bp["boxes"], MODELS):
        patch.set_facecolor(COL[m])
        patch.set_alpha(0.8)
        patch.set_edgecolor("white")
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xticks(range(1, len(MODELS) + 1))
    ax.set_xticklabels([LABEL[m] for m in MODELS])
    ax.set_ylim(-0.32, 0.62)
    for i, m in enumerate(MODELS, 1):
        ax.text(i, 0.565, f"{skill[m].median():.2f}", ha="center", va="center", fontsize=6.8, color=INK)
    ax.set_xlim(0.35, len(MODELS) + 0.5)
    ax.set_ylabel("Skill relative to persistence")
    ax.grid(axis="y")
    title(ax, "a", f"Skill at {len(w):,} stations")

    ax = axs[1]
    rows = [("ARIMA", "ARIMA", "ARIMA_MS", "multi-step")] + \
           [(m, m, f"{m}_level", "absolute level") for m in ("RF", "XGB", "LGBM", "LSTM")]
    y = np.arange(len(rows))[::-1]
    for yy, (name, good, bad, lab) in zip(y, rows):
        g, b = w[good].median(), w[bad].median()
        ax.plot([g, b], [yy, yy], color=GRID, lw=2.2, zorder=1, solid_capstyle="butt")
        ax.scatter(g, yy, s=26, color=COL[name], marker=MARK[name], zorder=3, edgecolor="white", linewidth=0.6)
        ax.scatter(b, yy, s=26, facecolor="white", edgecolor=COL[name], marker=MARK[name], zorder=3, linewidth=1.1)
        ax.text(b * 1.09, yy, lab, va="center", ha="left", fontsize=6.5, color=INK2)
    pm = w["Persistence"].median()
    ax.axvline(pm, color=COL["Persistence"], lw=0.7, ls="--")
    ax.text(pm * 1.04, len(rows) - 0.45, "persistence", fontsize=6.5, color=INK2, va="center")
    ax.set_yticks(y)
    ax.set_yticklabels([LABEL[r[0]] for r in rows])
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([0.04, 0.06, 0.1, 0.2, 0.4]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(["0.04", "0.06", "0.1", "0.2", "0.4"])
    ax.set_xlim(0.034, 0.9)
    ax.set_ylim(-0.6, len(rows) - 0.1)
    ax.set_xlabel("Median test RMSE (m, log scale)")
    ax.grid(axis="x")
    title(ax, "b", "Effect of evaluation protocol")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=4.5, color=INK2, label="this study"),
                       Line2D([], [], marker="o", ls="", ms=4.5, markerfacecolor="white", markeredgecolor=INK2,
                              label="protocol of earlier versions")],
              loc="center right", bbox_to_anchor=(1.0, 0.56), handletextpad=0.3, borderaxespad=0.2)
    save(fig, "Fig4_forecast_skill")


def fig5():
    w, _ = forecast_table()
    skill = 1 - w.div(w["Persistence"], axis=0)
    skill["basin"] = skill.index.str[:2]
    g = skill.groupby("basin")[MODELS].median()
    n = skill.groupby("basin").size()
    g = g.loc[[b for b in C.BASIN_NAMES if b in g.index]]
    fig, ax = plt.subplots(figsize=(120 * MM, 82 * MM), layout="constrained")
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", ["#f4f8fd", "#86b6ef", "#1c5cab"])
    im = ax.imshow(g.values, cmap=cmap, vmin=0, vmax=0.2, aspect="auto")
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            v = g.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.8, color="white" if v > 0.13 else INK)
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([LABEL[m] for m in MODELS])
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(g)))
    ax.set_yticklabels([f"{C.BASIN_NAMES[b]} ({n[b]})" for b in g.index])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.85, aspect=25, pad=0.02)
    cb.set_ticks([0, 0.05, 0.10, 0.15, 0.20])
    cb.set_ticklabels(["0.00", "0.05", "0.10", "0.15", "0.20"])
    cb.set_label("Median skill relative to persistence")
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=2)
    save(fig, "Fig5_basin_skill")


def fig6():
    w, status = forecast_table()
    skill = 1 - w.div(w["Persistence"], axis=0)
    ntr = status.loc[skill.index, "n_train"]
    bins = [100, 250, 500, 1000, 2500, 5000, 10000, 1e9]
    lab = ["100–\n250", "250–\n500", "500–\n1k", "1k–\n2.5k", "2.5k–\n5k", "5k–\n10k", ">10k"]
    cat = pd.cut(ntr, bins, labels=lab, right=False)
    counts = cat.value_counts().reindex(lab)
    fig, ax = plt.subplots(figsize=(120 * MM, 72 * MM), layout="constrained")
    for m in MODELS:
        g = skill[m].groupby(cat, observed=False).median()
        ax.plot(range(len(lab)), g.values, color=COL[m], marker=MARK[m], ms=3.5, lw=1.3, label=LABEL[m])
    ax.set_xticks(range(len(lab)))
    ax.set_xticklabels(lab)
    for i, c in enumerate(counts):
        ax.text(i, -0.024, f"n={c}", ha="center", va="center", fontsize=6.2, color=INK2)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_ylim(-0.032, 0.16)
    ax.set_yticks([0, 0.05, 0.10, 0.15])
    ax.set_yticklabels(["0.00", "0.05", "0.10", "0.15"])
    ax.set_xlabel("Training samples per station")
    ax.set_ylabel("Median skill relative to persistence")
    ax.grid(axis="y")
    ax.legend(loc="upper left", ncol=1, handlelength=1.6)
    save(fig, "Fig6_record_length")


def fig7():
    d = pd.read_csv(C.out("hyperparameter_runs.csv"), dtype={"Station_id": str})
    w = d.pivot_table(index="Station_id", columns=["model", "config"], values="RMSE")
    pers = w[("Persistence", "default")]
    ml = ["RF", "XGB", "LGBM", "LSTM"]
    cfgs = [("level", "absolute level", "o", False), ("default", "change-based, default", "s", True),
            ("tuned", "change-based, tuned", "D", True)]
    fig, axs = plt.subplots(1, 2, figsize=(FULL, 62 * MM), layout="constrained", gridspec_kw={"width_ratios": [1, 1]})
    ax = axs[0]
    for k, (cfg, lab, mk, filled) in enumerate(cfgs):
        for i, m in enumerate(ml):
            v = (1 - w[(m, cfg)] / pers).median()
            ax.scatter(i + (k - 1) * 0.22, v, s=24, marker=mk, color=COL[m] if filled else "white",
                       edgecolor=COL[m], linewidth=1.0, zorder=3)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xticks(range(len(ml)))
    ax.set_xticklabels([LABEL[m] for m in ml])
    ax.set_ylabel("Median skill relative to persistence")
    ax.grid(axis="y")
    title(ax, "a", "All three configurations")
    ax.legend(handles=[Line2D([], [], marker=mk, ls="", ms=4.5, markerfacecolor=INK2 if f else "white",
                              markeredgecolor=INK2, label=lab) for _, lab, mk, f in cfgs],
              loc="center left", bbox_to_anchor=(0.02, 0.38), handletextpad=0.3)
    ax = axs[1]
    for k, (cfg, lab, mk, filled) in enumerate(cfgs[1:], 1):
        for i, m in enumerate(ml):
            v = (1 - w[(m, cfg)] / pers).median()
            ax.scatter(i + (k - 1.5) * 0.25, v, s=26, marker=mk, color=COL[m], edgecolor="white", linewidth=0.5,
                       zorder=3)
    ar = pd.read_csv(C.out("forecast_metrics_long.csv"), dtype={"Station_id": str})
    ar = ar[(ar.model == "ARIMA") & ar.Station_id.isin(w.index)].set_index("Station_id").RMSE
    va = (1 - ar / pers.loc[ar.index]).median()
    ax.axhline(va, color=COL["ARIMA"], lw=0.8, ls="--")
    ax.text(3.45, va - 0.0025, "ARIMA", color=COL["ARIMA"], fontsize=6.5, ha="right", va="top")
    ax.set_xticks(range(len(ml)))
    ax.set_xticklabels([LABEL[m] for m in ml])
    ax.set_ylim(0.075, 0.125)
    ax.set_yticks([0.08, 0.09, 0.10, 0.11, 0.12])
    ax.set_yticklabels(["0.08", "0.09", "0.10", "0.11", "0.12"])
    ax.set_xlim(-0.5, 3.5)
    ax.grid(axis="y")
    title(ax, "b", "Change-based formulation (enlarged)")
    save(fig, "Fig7_hyperparameter")


def fig8():
    mk = pd.read_csv(C.out("trend_tests_stations.csv"), dtype={"Station_id": str})
    meta = pd.read_csv(C.out("station_metadata.csv"), dtype={"sid": str}).rename(columns={"sid": "Station_id"})
    t = mk[mk.tested].merge(meta[["Station_id", "lat", "lon"]], on="Station_id")
    for c in ("sig_bh", "sig_raw"):
        t[c] = t[c].astype(str).str.lower().eq("true")
    fig, axs = plt.subplots(1, 2, figsize=(FULL, 78 * MM), layout="constrained", gridspec_kw={"width_ratios": [1.45, 1]})
    ax = axs[0]
    ns = t[~t.sig_bh]
    ax.scatter(ns.lon, ns.lat, s=2.5, color=NS, lw=0, rasterized=True)
    inc = t[t.sig_bh & (t.tau > 0)]
    dec = t[t.sig_bh & (t.tau < 0)]
    ax.scatter(inc.lon, inc.lat, s=9, color=INC, marker="^", lw=0.3, edgecolor="white")
    ax.scatter(dec.lon, dec.lat, s=9, color=DEC, marker="v", lw=0.3, edgecolor="white")
    ax.set_aspect(1 / np.cos(np.deg2rad(55)))
    ax.set_xlim(-145, -50)
    ax.set_ylim(40, 84)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    title(ax, "a", f"Trends at {len(t):,} stations")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=3, color=NS, label=f"not significant ({len(ns):,})"),
                       Line2D([], [], marker="^", ls="", ms=4, color=INC, label=f"increasing ({len(inc)})"),
                       Line2D([], [], marker="v", ls="", ms=4, color=DEC, label=f"decreasing ({len(dec)})")],
              loc="upper right", handletextpad=0.2, borderaxespad=0.3)
    ax = axs[1]
    t["basin"] = t.Station_id.str[:2]
    g = t.groupby("basin").agg(raw=("sig_raw", "sum"), bh=("sig_bh", "sum")).reindex(list(C.BASIN_NAMES)).fillna(0)
    y = np.arange(len(g))[::-1]
    ax.barh(y + 0.19, g.raw, height=0.36, color="white", edgecolor=INC, lw=0.8, label="p < 0.05")
    ax.barh(y - 0.19, g.bh, height=0.36, color=INC, edgecolor=INC, lw=0.8, label="BH q < 0.05")
    ax.set_yticks(y)
    ax.set_yticklabels([C.BASIN_NAMES[b] for b in g.index])
    ax.set_xlabel("Stations with significant trend")
    ax.grid(axis="x")
    ax.legend(loc="lower right")
    title(ax, "b", "Before and after correction")
    save(fig, "Fig8_trends")


FIGS = {"fig3": fig3, "fig4": fig4, "fig5": fig5, "fig6": fig6, "fig7": fig7, "fig8": fig8}

if __name__ == "__main__":
    for name in (sys.argv[1:] or FIGS):
        FIGS[name]()
