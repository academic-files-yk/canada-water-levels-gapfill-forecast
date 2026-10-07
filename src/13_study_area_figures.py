"""
13 - Figures 1 and 2 of the manuscript: study area map and observation periods (vector PDF and 600 dpi PNG).

Fig. 1: stations of the archive on the eleven major drainage areas of Canada (Atlas of Canada, Natural Resources
        Canada, Open Government Licence - Canada), Canada Atlas Lambert projection (EPSG:3978); country and province
        outlines and lakes from Natural Earth (public domain).
Fig. 2: (a) periods with observations for every station, grouped by basin and ordered by the first year;
        (b) number of stations reporting per year, stacked by basin.
GIS inputs are read from GIS (default WORK/gis); download them with --download.
Output: figures/Fig1_study_area.*, figures/Fig2_observation_periods.*, output/observation_periods.csv
Usage : python 13_study_area_figures.py [--download] [fig1] [fig2]
"""
import glob
import importlib.util
import os
import sys
import urllib.request

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch
from pyproj import Transformer

import config as C

_spec = importlib.util.spec_from_file_location("figs", os.path.join(C.SRC, "12_figures.py"))
F = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(F)

GIS = os.environ.get("WL_GIS", os.path.join(C.WORK, "gis"))
SOURCES = {
    "major_drainage_areas.geojson": "https://geoappext.nrcan.gc.ca/arcgis/rest/services/NRCAN/AtlasWatershedsEN/MapServer/3/"
                                    "query?where=1%3D1&outFields=*&returnGeometry=true&outSR=4326"
                                    "&maxAllowableOffset=0.01&f=geojson",
    "ne_50m_admin_0_countries.zip": "https://naciscdn.org/naturalearth/50m/cultural/ne_50m_admin_0_countries.zip",
    "ne_50m_admin_1.zip": "https://naciscdn.org/naturalearth/50m/cultural/ne_50m_admin_1_states_provinces_lakes.zip",
    "ne_50m_lakes.zip": "https://naciscdn.org/naturalearth/50m/physical/ne_50m_lakes.zip",
}
CRS = "EPSG:3978"
BASINS = [f"{i:02d}" for i in range(1, 12)]
TINT = {"01": "#cdb9dc", "02": "#a9cfe6", "03": "#cbe7c3", "04": "#fbd3a6", "05": "#f2b8b4", "06": "#b6dfd9",
        "07": "#e8d6a6", "08": "#a9d4a0", "09": "#f5c9de", "10": "#cfd7e6", "11": "#efe08f"}
DOT, DOT_OTHER, LAND, LINE = "#1f2a33", "#8a8a8a", "#efefef", "#5a5a5a"


def download():
    os.makedirs(GIS, exist_ok=True)
    for name, url in SOURCES.items():
        path = os.path.join(GIS, name)
        if not os.path.exists(path):
            print("downloading", name)
            urllib.request.urlretrieve(url, path)


def observation_periods():
    """Runs of consecutive observed days for every station of the raw archive (cached)."""
    path = C.out("observation_periods.csv")
    if os.path.exists(path):
        return pd.read_csv(path, dtype={"Station_id": str, "basin": str}, parse_dates=["start", "end"])
    files = sorted(glob.glob(os.path.join(C.RAW, "*.parquet"))) if os.path.isdir(C.RAW) else [C.RAW]
    rows = []
    for f in files:
        d = pd.read_parquet(f, columns=["Station ID", "Date", "Value", "Basin"])
        d = pd.DataFrame({"Station ID": d["Station ID"].astype(str).to_numpy(),
                          "Date": pd.to_datetime(d["Date"].astype(str), errors="coerce").to_numpy(),
                          "Value": pd.to_numeric(d["Value"], errors="coerce").to_numpy()})
        d = d.dropna(subset=["Value", "Date"]).drop_duplicates(["Station ID", "Date"])
        d = d.sort_values(["Station ID", "Date"])
        new = (d["Station ID"] != d["Station ID"].shift()) | (d["Date"].diff().dt.days != 1)
        d["run"] = new.to_numpy(dtype=bool).cumsum()
        g = d.groupby("run").agg(Station_id=("Station ID", "first"), start=("Date", "first"), end=("Date", "last"))
        rows.append(g)
    p = pd.concat(rows, ignore_index=True)
    p["basin"] = p.Station_id.str[:2]
    p.to_csv(path, index=False)
    return p


def fig1():
    meta = pd.read_csv(C.out("station_metadata.csv"), dtype={"sid": str, "basin": str})
    used = set(pd.read_csv(C.out("forecast_status.csv"), dtype={"Station_id": str}).query("status == 'ok'").Station_id)
    mda = gpd.read_file(os.path.join(GIS, "major_drainage_areas.geojson"))
    mda = mda[mda.OBJECTID_1 <= 11].copy()
    mda["basin"] = mda.OBJECTID_1.map(lambda i: f"{int(i):02d}")
    mda = mda.to_crs(CRS)
    world = gpd.read_file("zip://" + os.path.join(GIS, "ne_50m_admin_0_countries.zip")).to_crs(CRS)
    prov = gpd.read_file("zip://" + os.path.join(GIS, "ne_50m_admin_1.zip"))
    prov = prov[prov.adm0_a3 == "CAN"].to_crs(CRS)
    lakes = gpd.read_file("zip://" + os.path.join(GIS, "ne_50m_lakes.zip")).to_crs(CRS)
    tr = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    x, y = tr.transform(meta.lon.values, meta.lat.values)
    meta["x"], meta["y"] = x, y

    fig = plt.figure(figsize=(F.FULL, 104 * F.MM))
    ax = fig.add_axes([0.0, 0.0, 0.70, 1.0])
    xmin, xmax, ymin, ymax = -2.45e6, 3.05e6, -0.85e6, 4.35e6
    ax.set_facecolor("white")
    world.plot(ax=ax, color=LAND, edgecolor="#bdbdbd", lw=0.3)
    for b in BASINS:
        mda[mda.basin == b].plot(ax=ax, color=TINT[b], edgecolor="none")
    prov.boundary.plot(ax=ax, color="white", lw=0.35)
    mda.boundary.plot(ax=ax, color=LINE, lw=0.3)
    lakes[lakes.scalerank <= 3].plot(ax=ax, color="#dfeaf3", edgecolor="#9fb7c9", lw=0.2)
    other = meta[~meta.sid.isin(used)]
    main = meta[meta.sid.isin(used)]
    ax.scatter(other.x, other.y, s=1.6, color=DOT_OTHER, lw=0, zorder=4)
    ax.scatter(main.x, main.y, s=1.6, color=DOT, lw=0, zorder=5)

    # graticule with labels on the left and bottom edges
    lons, lats = np.arange(-140, -39, 20), np.arange(40, 81, 10)
    for lon in lons:
        la = np.linspace(30, 88, 300)
        gx, gy = tr.transform(np.full_like(la, lon), la)
        ax.plot(gx, gy, color="#9e9e9e", lw=0.3, ls=(0, (2, 2)), zorder=3)
        k = np.where(np.diff(np.sign(gy - ymin)))[0]
        if len(k) and xmin < gx[k[0]] < xmax:
            ax.text(gx[k[0]], ymin - 0.04e6, f"{-lon}°W", ha="center", va="top", fontsize=6.5, color=F.INK2)
    for lat in lats:
        lo = np.linspace(-175, -20, 400)
        gx, gy = tr.transform(lo, np.full_like(lo, lat))
        ax.plot(gx, gy, color="#9e9e9e", lw=0.3, ls=(0, (2, 2)), zorder=3)
        k = np.where(np.diff(np.sign(gx - xmin)))[0]
        if len(k) and ymin < gy[k[0]] < ymax:
            ax.text(xmin - 0.04e6, gy[k[0]], f"{lat}°N", ha="right", va="center", fontsize=6.5, color=F.INK2)

    # basin codes
    for _, r in mda.iterrows():
        geom = max(r.geometry.geoms, key=lambda g: g.area) if r.geometry.geom_type == "MultiPolygon" else r.geometry
        p = geom.representative_point()
        dx, dy = {"01": (1.5e5, -1.5e5), "11": (0, -1.2e5), "04": (0.5e5, 0.6e5)}.get(r.basin, (0, 0))
        ax.text(p.x + dx, p.y + dy, r.basin, ha="center", va="center", fontsize=7.5, fontweight="bold", color=F.INK,
                zorder=6, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))

    # scale bar (true scale near the standard parallels of the projection)
    sx, sy = 1.55e6, -0.62e6
    for i, c in enumerate(["#333333", "white", "#333333", "white"]):
        ax.add_patch(plt.Rectangle((sx + i * 250e3, sy), 250e3, 45e3, fc=c, ec="#333333", lw=0.4, zorder=7))
    for v, lab in ((0, "0"), (500, "500"), (1000, "1000 km")):
        ax.text(sx + v * 1e3, sy + 70e3, lab, ha="center", va="bottom", fontsize=6.5, color=F.INK2, zorder=7)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(F.INK2)
        s.set_linewidth(0.6)
    ax.set_position([0.07, 0.065, 0.61, 0.925])

    n = meta.groupby("basin").size()
    handles = [Patch(fc=TINT[b], ec=LINE, lw=0.4, label=f"{b}  {C.BASIN_NAMES[b]} ({n.get(b, 0):,})") for b in BASINS]
    handles += [plt.Line2D([], [], ls="", marker="o", ms=2.6, color=DOT, label=f"Station in forecasting analysis ({len(main):,})"),
                plt.Line2D([], [], ls="", marker="o", ms=2.6, color=DOT_OTHER, label=f"Other station ({len(other):,})")]
    leg = fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.69, 0.52), fontsize=6.6, handlelength=1.4,
                     handleheight=1.0, labelspacing=0.55, title="Drainage area (stations)", title_fontsize=7,
                     alignment="left")
    leg._legend_box.align = "left"
    F.save(fig, "Fig1_study_area")


def yearly_fraction(p, stations, years):
    """Fraction of the days of each year with an observation (stations x years)."""
    idx = pd.Series(np.arange(len(stations)), index=stations)
    M = np.zeros((len(stations), len(years)))
    y0 = years[0]
    for sid, a, b in zip(p.Station_id, p.start, p.end):
        r = idx[sid]
        for y in range(a.year, b.year + 1):
            lo = max(a, pd.Timestamp(y, 1, 1))
            hi = min(b, pd.Timestamp(y, 12, 31))
            if y0 <= y <= years[-1]:
                M[r, y - y0] += (hi - lo).days + 1
    ndays = np.array([366 if pd.Timestamp(y, 1, 1).is_leap_year else 365 for y in years])
    return M / ndays


def fig2():
    from matplotlib.colors import LinearSegmentedColormap
    p = observation_periods()
    first = p.groupby("Station_id").start.min()
    order = (pd.DataFrame({"first": first, "basin": first.index.str[:2]})
             .sort_values(["basin", "first"]))
    stations = order.index.to_numpy()
    years = np.arange(1850, 2025)
    M = yearly_fraction(p, stations, years)
    last = int(years[M.sum(axis=0) > 0].max())

    fig = plt.figure(figsize=(F.FULL, 150 * F.MM), layout="constrained")
    gs = fig.add_gridspec(2, 1, height_ratios=[2.4, 1])
    ax = fig.add_subplot(gs[0])
    n = len(stations)
    ticks, labels = [], []
    for b in BASINS:
        rows = np.where(order.basin.to_numpy() == b)[0]
        lo, hi = rows.min() - 0.5, rows.max() + 0.5
        ax.axhspan(lo, hi, color=TINT[b], alpha=0.45, lw=0, zorder=0)
        if lo > 0:
            ax.axhline(lo, color="white", lw=0.9, zorder=3)
        ticks.append((lo + hi) / 2)
        labels.append(f"{b} {C.BASIN_NAMES[b]}")
    cmap = LinearSegmentedColormap.from_list("obs", ["#9fb0c2", "#14202b"])
    im = ax.imshow(np.ma.masked_equal(M, 0), cmap=cmap, vmin=0, vmax=1, aspect="auto", origin="upper",
                   extent=(years[0], years[-1] + 1, n - 0.5, -0.5), interpolation="antialiased", zorder=2)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xlim(1850, last + 1)
    ax.set_xticks(np.arange(1850, 2026, 25))
    ax.set_xlabel("Year")
    ax.spines["left"].set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.45, aspect=18, pad=0.01)
    cb.set_label("Fraction of days with observations", fontsize=7)
    cb.outline.set_linewidth(0.4)
    F.title(ax, "a", f"Observation record of each station ({n:,} stations; one row per station)")

    ax = fig.add_subplot(gs[1])
    yy = years[years < last]          # the last year of the archive is incomplete
    base = np.zeros(len(yy))
    for b in BASINS:
        c = (M[order.basin.to_numpy() == b][:, :len(yy)] > 0).sum(axis=0)
        ax.fill_between(yy + 0.5, base, base + c, color=TINT[b], ec="white", lw=0.3, step="mid", label=b)
        base = base + c
    ax.plot(yy + 0.5, base, color=DOT, lw=0.8, drawstyle="steps-mid")
    ax.set_xlim(1850, last + 1)
    ax.set_xticks(np.arange(1850, 2026, 25))
    ax.set_ylim(0, base.max() * 1.1)
    ax.set_xlabel("Year")
    ax.set_ylabel("Stations reporting")
    ax.grid(axis="y")
    ax.legend(ncol=11, loc="upper left", fontsize=6.5, handlelength=1.0, columnspacing=0.8, handletextpad=0.4,
              title="Drainage area", title_fontsize=6.5, alignment="left")
    F.title(ax, "b", f"Number of stations with at least one observation per year (1850–{last - 1})")
    F.save(fig, "Fig2_observation_periods")
    print("last year in archive:", last, "| max stations in one year:", int(base.max()), "in", yy[base.argmax()],
          "| stations before 1900:", int((M[:, :50].sum(axis=1) > 0).sum()))


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--download" in args:
        download()
    todo = [a for a in args if a.startswith("fig")] or ["fig1", "fig2"]
    for name in todo:
        globals()[name]()
