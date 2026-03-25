from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html


HERE = Path(__file__).resolve().parent

STATS_FILE_OPTIONS = ["stats_by_depth_all.csv", "stats_by_depth_all_v2.csv"]
STATS_PATHS = [HERE / name for name in STATS_FILE_OPTIONS if (HERE / name).exists()]
if not STATS_PATHS:
    raise FileNotFoundError(f"No stats CSVs found. Expected one of: {STATS_FILE_OPTIONS} in {HERE}")

DEFAULT_STATS_FILE = STATS_PATHS[0].name


REQUIRED_COLS = {
    "variable",
    "station_type",
    "month",
    "depth_bin_low",
    "depth_bin_high",
    "n",
    "median",
    "p01",
    "p99",
    "min",
    "max",
}


def _load_df(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["month"] = pd.to_numeric(df["month"], errors="coerce").astype("Int64")
    df["depth_bin_low"] = pd.to_numeric(df["depth_bin_low"], errors="coerce")
    df["depth_bin_high"] = pd.to_numeric(df["depth_bin_high"], errors="coerce")
    for c in ["median", "p01", "p99", "min", "max", "n"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(
        subset=[
            "variable",
            "station_type",
            "month",
            "depth_bin_low",
            "depth_bin_high",
            "median",
            "p01",
            "p99",
            "min",
            "max",
        ]
    )
    return df


_STATS_CACHE: dict[str, pd.DataFrame] = {}


def _get_stats_df(stats_file: str) -> pd.DataFrame:
    # Only allow known filenames (avoid arbitrary reads via URL params etc.)
    if stats_file not in STATS_FILE_OPTIONS:
        stats_file = DEFAULT_STATS_FILE
    path = HERE / stats_file
    if not path.exists():
        # Fall back to default available file.
        path = HERE / DEFAULT_STATS_FILE
        stats_file = DEFAULT_STATS_FILE
    if stats_file not in _STATS_CACHE:
        _STATS_CACHE[stats_file] = _load_df(path)
    return _STATS_CACHE[stats_file]


def _stats_meta(df: pd.DataFrame) -> tuple[list[str], list[str], list[int]]:
    variables = sorted(df["variable"].unique().tolist())
    station_types = sorted(df["station_type"].unique().tolist())
    months = sorted(df["month"].dropna().unique().astype(int).tolist())
    return variables, station_types, months

STATIONS_CSV = HERE / "Complete_Station_Mean_Coords.csv"
if not STATIONS_CSV.exists():
    # Backward-compatible fallback
    STATIONS_CSV = HERE / "Station_Mean_Coords.csv"
POLYGON_GEOJSON_PATH = HERE / "polygon_shallow_lte15m_SouthFlorida.geojson"
STATION_CLASSIFICATION_JSON = HERE / "station_depth_classification.json"


def _load_stations(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"station", "lat_mean", "lon_mean"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Stations CSV is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["lat_mean"] = pd.to_numeric(df["lat_mean"], errors="coerce")
    df["lon_mean"] = pd.to_numeric(df["lon_mean"], errors="coerce")
    df["station"] = df["station"].astype(str)
    df = df.dropna(subset=["lat_mean", "lon_mean", "station"])

    # Heuristic fix for occasional swapped lon/lat rows.
    # Most stations here are expected to be in the N hemisphere and W longitudes.
    lat_med = float(df["lat_mean"].median())
    lon_med = float(df["lon_mean"].median())
    expected_lat_positive = lat_med >= 0
    expected_lon_negative = lon_med <= 0

    lat = df["lat_mean"]
    lon = df["lon_mean"]
    looks_swapped = ((lat < 0) == expected_lat_positive) & ((lon > 0) == expected_lon_negative)
    if looks_swapped.any():
        df.loc[looks_swapped, ["lat_mean", "lon_mean"]] = df.loc[looks_swapped, ["lon_mean", "lat_mean"]].to_numpy()

    return df


def _load_geojson(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


STATIONS_DF = _load_stations(STATIONS_CSV) if STATIONS_CSV.exists() else pd.DataFrame(columns=["station", "lat_mean", "lon_mean"])
POLYGON_GEOJSON = _load_geojson(POLYGON_GEOJSON_PATH) if POLYGON_GEOJSON_PATH.exists() else None


def _get_prepared_shallow_geom():
    """
    Returns (hit_test_fn, info_message).

    The returned function tests whether a (lon, lat) point intersects the shallow polygon
    union (intersects is more forgiving than strict contains and includes boundary points).
    """
    if not isinstance(POLYGON_GEOJSON, dict) or POLYGON_GEOJSON.get("type") != "FeatureCollection":
        return None, "Polygon GeoJSON missing or invalid."

    try:
        from shapely.geometry import Point, shape
        from shapely.ops import unary_union
        from shapely.prepared import prep
    except Exception as e:
        return None, f"Shapely unavailable ({type(e).__name__})."

    feats = POLYGON_GEOJSON.get("features") or []
    geoms = []
    for ft in feats:
        geom = ft.get("geometry")
        if not geom:
            continue
        try:
            geoms.append(shape(geom))
        except Exception:
            continue

    if not geoms:
        return None, "No polygon geometries found in GeoJSON."

    try:
        union = unary_union(geoms)
        prepared = prep(union)
        minx, miny, maxx, maxy = union.bounds
    except Exception as e:
        return None, f"Failed to build polygon union ({type(e).__name__})."

    def hit(lon: float, lat: float) -> bool:
        # Shapely Point is (x, y) == (lon, lat)
        return bool(prepared.intersects(Point(float(lon), float(lat))))

    info = f"Polygon bounds lon[{minx:.3f},{maxx:.3f}] lat[{miny:.3f},{maxy:.3f}]"
    return hit, info


_SHALLOW_HIT, _SHALLOW_INFO = _get_prepared_shallow_geom()


def _auto_zoom(lat: list[float], lon: list[float]) -> int:
    if not lat or not lon:
        return 6
    lat_range = max(lat) - min(lat)
    lon_range = max(lon) - min(lon)
    r = max(lat_range, lon_range)
    if r > 20:
        return 3
    if r > 10:
        return 4
    if r > 5:
        return 5
    if r > 2:
        return 6
    if r > 1:
        return 7
    if r > 0.5:
        return 8
    return 9


def _fig_station_map() -> tuple[go.Figure, str]:
    if STATIONS_DF.empty:
        fig = go.Figure()
        fig.update_layout(template="plotly_white", height=650, margin=dict(l=10, r=10, t=30, b=10))
        return fig, f"Missing or empty stations file: {STATIONS_CSV.name}"

    lats = STATIONS_DF["lat_mean"].astype(float).tolist()
    lons = STATIONS_DF["lon_mean"].astype(float).tolist()
    center = {"lat": float(sum(lats) / len(lats)), "lon": float(sum(lons) / len(lons))}
    zoom = _auto_zoom(lats, lons)

    fig = go.Figure()

    # Polygon overlay (if available)
    if isinstance(POLYGON_GEOJSON, dict) and POLYGON_GEOJSON.get("type") == "FeatureCollection":
        feats = POLYGON_GEOJSON.get("features") or []
        locations = []
        for i, ft in enumerate(feats):
            locations.append(ft.get("id") or str(i))

        # Ensure each feature has an id so Plotly can match `locations`.
        for i, ft in enumerate(feats):
            if "id" not in ft or ft["id"] is None:
                ft["id"] = str(i)

        fig.add_trace(
            go.Choroplethmapbox(
                geojson=POLYGON_GEOJSON,
                locations=locations,
                z=[1] * len(locations),
                showscale=False,
                colorscale=[[0, "rgba(0, 123, 255, 0.35)"], [1, "rgba(0, 123, 255, 0.35)"]],
                marker_opacity=0.45,
                marker_line_width=2,
                marker_line_color="rgba(0, 123, 255, 0.90)",
                hovertemplate="Shallow polygon<extra></extra>",
                name="Shallow polygon",
            )
        )

    stations = STATIONS_DF.copy()
    if _SHALLOW_HIT is not None:
        stations["in_polygon"] = [
            bool(_SHALLOW_HIT(lon, lat))
            for lon, lat in zip(stations["lon_mean"].astype(float), stations["lat_mean"].astype(float))
        ]
    else:
        stations["in_polygon"] = False

    inside = stations[stations["in_polygon"]].copy()
    outside = stations[~stations["in_polygon"]].copy()

    # Export station classification for downstream use.
    if _SHALLOW_HIT is not None:
        shallow = (
            stations.loc[stations["in_polygon"], "station"].astype(str).dropna().drop_duplicates().sort_values().tolist()
        )
        deep = (
            stations.loc[~stations["in_polygon"], "station"].astype(str).dropna().drop_duplicates().sort_values().tolist()
        )
        payload = {"shallow": shallow, "deep": deep}
        try:
            STATION_CLASSIFICATION_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        except Exception:
            # Non-fatal; map should still render.
            pass

    # Station points (outside first so inside is on top)
    if not outside.empty:
        fig.add_trace(
            go.Scattermapbox(
                lat=outside["lat_mean"].astype(float).tolist(),
                lon=outside["lon_mean"].astype(float).tolist(),
                mode="markers",
                marker=dict(size=8, color="rgba(120, 120, 120, 0.75)"),
                text=outside["station"].tolist(),
                hovertemplate="Station %{text}<br>Outside polygon<br>(%{lat:.4f}, %{lon:.4f})<extra></extra>",
                name="Stations (outside)",
            )
        )

    if not inside.empty:
        fig.add_trace(
            go.Scattermapbox(
                lat=inside["lat_mean"].astype(float).tolist(),
                lon=inside["lon_mean"].astype(float).tolist(),
                mode="markers",
                marker=dict(size=10, color="rgba(220, 20, 60, 0.92)"),
                text=inside["station"].tolist(),
                hovertemplate="Station %{text}<br><b>Inside polygon</b><br>(%{lat:.4f}, %{lon:.4f})<extra></extra>",
                name="Stations (inside)",
            )
        )

    fig.update_layout(
        title=dict(
            text="Stations and shallow polygon",
            y=0.99,
            yanchor="top",
            pad=dict(b=10),
        ),
        height=650,
        margin=dict(l=10, r=10, t=70, b=10),
        # Use a minimal raster basemap without admin boundary clutter.
        mapbox=dict(
            style="white-bg",
            center=center,
            zoom=zoom,
            layers=[
                {
                    "sourcetype": "raster",
                    "source": ["https://basemaps.cartocdn.com/rastertiles/light_nolabels/{z}/{x}/{y}.png"],
                    "below": "traces",
                }
            ],
        ),
        legend=dict(orientation="h", yanchor="bottom", y=0.965, xanchor="left", x=0),
    )
    if _SHALLOW_HIT is None:
        status = f"Inside/outside unavailable. {_SHALLOW_INFO}"
    else:
        status = (
            f"Stations inside polygon: {len(inside)} | outside: {len(outside)}. "
            f"Wrote: {STATION_CLASSIFICATION_JSON.name}. {_SHALLOW_INFO}"
        )
    return fig, status


def _subset_depth_on_x(df: pd.DataFrame, var: str, station_type: str, month: int) -> pd.DataFrame:
    sub = df[(df["variable"] == var) & (df["station_type"] == station_type) & (df["month"] == month)].copy()
    return sub.sort_values(["depth_bin_low", "depth_bin_high"])


def _depth_low_labels(sub: pd.DataFrame) -> list[str]:
    return [f"{v:g}" for v in sub["depth_bin_low"].to_list()]


def _hover_text_depth(sub: pd.DataFrame) -> list[str]:
    parts = zip(
        sub["depth_bin_low"],
        sub["depth_bin_high"],
        sub["n"],
        sub["min"],
        sub["p01"],
        sub["median"],
        sub["p99"],
        sub["max"],
    )
    return [
        (
            f"Depth bin: {low:g}–{high:g}<br>"
            f"n={n:.0f}<br>"
            f"min={mn:.4g}<br>"
            f"p01={p01:.4g}<br>"
            f"median={med:.4g}<br>"
            f"p99={p99:.4g}<br>"
            f"max={mx:.4g}"
        )
        for low, high, n, mn, p01, med, p99, mx in parts
    ]


def _fig_depth_on_x(df: pd.DataFrame, var: str, station_type: str, month: int) -> tuple[go.Figure, str]:
    sub = _subset_depth_on_x(df, var, station_type, month)
    title = f"{var} | {station_type} | month={month}"

    if sub.empty:
        fig = go.Figure()
        fig.update_layout(
            title=title + " (no data)",
            template="plotly_white",
            height=600,
            margin=dict(l=60, r=30, t=60, b=60),
        )
        return fig, "No data for this selection."

    xlabels = _depth_low_labels(sub)
    trace = go.Box(
        x=xlabels,
        q1=sub["p01"],
        median=sub["median"],
        q3=sub["p99"],
        lowerfence=sub["min"],
        upperfence=sub["max"],
        boxpoints=False,
        text=_hover_text_depth(sub),
        hovertemplate="%{text}<extra></extra>",
    )

    fig = go.Figure(data=[trace])
    fig.update_layout(
        title=title,
        xaxis_title="depth_bin_low",
        xaxis=dict(type="category", categoryorder="array", categoryarray=xlabels),
        yaxis_title=var,
        template="plotly_white",
        height=600,
        margin=dict(l=60, r=30, t=60, b=60),
    )
    return fig, ""


@dataclass(frozen=True)
class DepthBin:
    low: float
    high: float

    @property
    def value(self) -> str:
        return f"{self.low}|{self.high}"

    @property
    def label(self) -> str:
        return f"{self.low:g}–{self.high:g}"

    @staticmethod
    def from_value(v: str) -> "DepthBin":
        low_s, high_s = v.split("|", 1)
        return DepthBin(float(low_s), float(high_s))


def _depth_bin_options(df: pd.DataFrame, var: str, station_type: str) -> list[DepthBin]:
    bins = (
        df[(df["variable"] == var) & (df["station_type"] == station_type)][["depth_bin_low", "depth_bin_high"]]
        .drop_duplicates()
        .dropna()
        .sort_values(["depth_bin_low", "depth_bin_high"])
    )
    out: list[DepthBin] = []
    for low, high in bins.itertuples(index=False, name=None):
        out.append(DepthBin(float(low), float(high)))
    return out


def _subset_month_on_x(df: pd.DataFrame, var: str, station_type: str, depth_bin: DepthBin) -> pd.DataFrame:
    sub = df[
        (df["variable"] == var)
        & (df["station_type"] == station_type)
        & (df["depth_bin_low"] == depth_bin.low)
        & (df["depth_bin_high"] == depth_bin.high)
    ].copy()
    return sub.sort_values(["month"])


def _hover_text_month(sub: pd.DataFrame) -> list[str]:
    parts = zip(
        sub["month"].astype(int),
        sub["n"],
        sub["min"],
        sub["p01"],
        sub["median"],
        sub["p99"],
        sub["max"],
    )
    return [
        (
            f"Month: {m}<br>"
            f"n={n:.0f}<br>"
            f"min={mn:.4g}<br>"
            f"p01={p01:.4g}<br>"
            f"median={med:.4g}<br>"
            f"p99={p99:.4g}<br>"
            f"max={mx:.4g}"
        )
        for m, n, mn, p01, med, p99, mx in parts
    ]


def _fig_month_on_x(df: pd.DataFrame, months: list[int], var: str, station_type: str, depth_bin: DepthBin) -> tuple[go.Figure, str]:
    sub = _subset_month_on_x(df, var, station_type, depth_bin)
    title = f"{var} | {station_type} | depth={depth_bin.label}"

    if sub.empty:
        fig = go.Figure()
        fig.update_layout(
            title=title + " (no data)",
            template="plotly_white",
            height=600,
            margin=dict(l=60, r=30, t=60, b=60),
        )
        return fig, "No data for this selection."

    trace = go.Box(
        x=sub["month"].astype(int),
        q1=sub["p01"],
        median=sub["median"],
        q3=sub["p99"],
        lowerfence=sub["min"],
        upperfence=sub["max"],
        boxpoints=False,
        text=_hover_text_month(sub),
        hovertemplate="%{text}<extra></extra>",
    )

    fig = go.Figure(data=[trace])
    fig.update_layout(
        title=title,
        xaxis_title="month",
        xaxis=dict(type="category", categoryorder="array", categoryarray=months),
        yaxis_title=var,
        template="plotly_white",
        height=600,
        margin=dict(l=60, r=30, t=60, b=60),
    )
    return fig, ""


@dataclass(frozen=True)
class SeasonSegment:
    start_month: int
    end_month: int
    lower: float
    upper: float

    @property
    def label(self) -> str:
        return f"{self.start_month}–{self.end_month}: [{self.lower:.4g}, {self.upper:.4g}]"


def _season_partition_cost(q1: list[float], q3: list[float], segments: list[tuple[int, int]]) -> float:
    total = 0.0
    for i, j in segments:
        seg_q1 = q1[i : j + 1]
        seg_q3 = q3[i : j + 1]
        if not seg_q1 or not seg_q3:
            continue
        low = min(seg_q1)
        high = max(seg_q3)
        total += sum(v - low for v in seg_q1) + sum(high - v for v in seg_q3)
    return float(total)


def _optimal_season_segments_linear(months: list[int], q1: list[float], q3: list[float], seasons: int) -> tuple[list[tuple[int, int]], float]:
    """
    Partition the month indices into `seasons` contiguous segments to minimize:
      sum_k (q1_k - min(q1 in seg(k))) + (max(q3 in seg(k)) - q3_k)
    With the implied "cover" limits:
      lower = min(q1 in segment), upper = max(q3 in segment)
    """
    n = len(months)
    if n == 0:
        return [], 0.0
    seasons = max(1, min(int(seasons), 4, n))

    # Precompute segment costs.
    cost = [[0.0] * n for _ in range(n)]
    for i in range(n):
        min_q1 = q1[i]
        max_q3 = q3[i]
        sum_q1 = 0.0
        sum_q3 = 0.0
        for j in range(i, n):
            min_q1 = min(min_q1, q1[j])
            max_q3 = max(max_q3, q3[j])
            sum_q1 += q1[j]
            sum_q3 += q3[j]
            seg_len = j - i + 1
            cost[i][j] = (sum_q1 - seg_len * min_q1) + (seg_len * max_q3 - sum_q3)

    # DP: dp[s][j] = min cost to partition 0..j into s segments
    INF = 1e300
    dp = [[INF] * n for _ in range(seasons + 1)]
    prev = [[-1] * n for _ in range(seasons + 1)]

    for j in range(n):
        dp[1][j] = cost[0][j]
        prev[1][j] = -1

    for s in range(2, seasons + 1):
        for j in range(n):
            best = INF
            best_i = -1
            # last segment starts at i (>= s-1)
            for i in range(s - 1, j + 1):
                v = dp[s - 1][i - 1] + cost[i][j] if i - 1 >= 0 else INF
                if v < best:
                    best = v
                    best_i = i
            dp[s][j] = best
            prev[s][j] = best_i

    # Reconstruct.
    s = seasons
    j = n - 1
    segments: list[tuple[int, int]] = []
    while s >= 1 and j >= 0:
        i = prev[s][j]
        if s == 1:
            segments.append((0, j))
            break
        if i < 0:
            # Shouldn't happen, but fall back to a single segment.
            segments = [(0, n - 1)]
            break
        segments.append((i, j))
        j = i - 1
        s -= 1
    segments.reverse()
    return segments, _season_partition_cost(q1, q3, segments)


def _optimal_season_month_groups_wrap(months: list[int], q1: list[float], q3: list[float], seasons: int) -> list[list[int]]:
    """
    Like the linear optimizer, but months are on a circle and segments may wrap across Dec->Jan.

    Strategy: try every rotation of the observed months, solve the linear DP, and pick the
    lowest-cost solution. n is small (<= 12), so this is fast.
    """
    n = len(months)
    if n == 0:
        return []
    seasons = max(1, min(int(seasons), 4, n))

    best_cost = 1e300
    best_groups: list[list[int]] = []

    for r in range(n):
        months_r = months[r:] + months[:r]
        q1_r = q1[r:] + q1[:r]
        q3_r = q3[r:] + q3[:r]

        segs, cost = _optimal_season_segments_linear(months_r, q1_r, q3_r, seasons=seasons)
        if cost < best_cost:
            best_cost = cost
            best_groups = [months_r[i : j + 1] for (i, j) in segs]

    return best_groups


def _add_season_limits(fig: go.Figure, sub: pd.DataFrame, seasons: int) -> tuple[go.Figure, str]:
    if sub.empty:
        return fig, ""

    s = max(1, min(int(seasons), 4))
    sub2 = sub.dropna(subset=["month", "p01", "p99"]).copy()
    if sub2.empty:
        return fig, ""

    sub2["month"] = sub2["month"].astype(int)
    sub2 = sub2.sort_values(["month"])

    months = sub2["month"].astype(int).to_list()
    q1 = sub2["p01"].astype(float).to_list()
    q3 = sub2["p99"].astype(float).to_list()
    if len(months) < 2:
        return fig, ""

    groups = _optimal_season_month_groups_wrap(months, q1, q3, seasons=s)
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    segments: list[SeasonSegment] = []
    for k, seg_months in enumerate(groups):
        if not seg_months:
            continue

        # Compute limits using the underlying q1/q3 arrays for these months.
        idx = [i for i, m in enumerate(months) if m in set(seg_months)]
        seg_q1 = [q1[i] for i in idx]
        seg_q3 = [q3[i] for i in idx]
        if not seg_q1 or not seg_q3:
            continue

        low = float(min(seg_q1))
        high = float(max(seg_q3))
        segments.append(
            SeasonSegment(start_month=int(seg_months[0]), end_month=int(seg_months[-1]), lower=low, upper=high)
        )

        # Draw as one or two traces if it wraps, to avoid a diagonal jump from Dec to Jan.
        color = palette[k % len(palette)]
        blocks: list[list[int]] = []
        current: list[int] = []
        prev_m: int | None = None
        for m in seg_months:
            if prev_m is not None and m < prev_m:
                if current:
                    blocks.append(current)
                current = [m]
            else:
                current.append(m)
            prev_m = m
        if current:
            blocks.append(current)

        for bi, x in enumerate(blocks):
            showlegend = bi == 0
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=[low] * len(x),
                    mode="lines",
                    line=dict(color=color, width=2, dash="dash"),
                    name=f"Season {k + 1} lower",
                    legendgroup=f"season{k + 1}",
                    showlegend=showlegend,
                    hovertemplate=f"Season {k + 1} lower<br>months {segments[-1].start_month}–{segments[-1].end_month}<br>y={low:.4g}<extra></extra>",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=[high] * len(x),
                    mode="lines",
                    line=dict(color=color, width=2, dash="dash"),
                    name=f"Season {k + 1} upper",
                    legendgroup=f"season{k + 1}",
                    showlegend=False,
                    hovertemplate=f"Season {k + 1} upper<br>months {segments[-1].start_month}–{segments[-1].end_month}<br>y={high:.4g}<extra></extra>",
                )
            )

    season_text = "; ".join([seg.label for seg in segments])
    return fig, f"Season limits ({s}): {season_text}"


def _fixed_season_limits(sub: pd.DataFrame) -> tuple[SeasonSegment | None, SeasonSegment | None]:
    """
    Fixed seasons:
      - Season A: months 12 through 5 (wrap) => {12,1,2,3,4,5}
      - Season B: months 6 through 11        => {6,7,8,9,10,11}
    Limits cover q1/q3 where q1=p01 and q3=p99.
    """
    if sub.empty:
        return None, None

    sub2 = sub.dropna(subset=["month", "p01", "p99"]).copy()
    if sub2.empty:
        return None, None

    sub2["month"] = sub2["month"].astype(int)
    season_a = {12, 1, 2, 3, 4, 5}
    season_b = {6, 7, 8, 9, 10, 11}

    def seg(month_set: set[int], start: int, end: int) -> SeasonSegment | None:
        ss = sub2[sub2["month"].isin(month_set)]
        if ss.empty:
            return None
        low = float(ss["p01"].min())
        high = float(ss["p99"].max())
        return SeasonSegment(start_month=start, end_month=end, lower=low, upper=high)

    return seg(season_a, 12, 5), seg(season_b, 6, 11)


def _add_fixed_season_limits(fig: go.Figure, sub: pd.DataFrame) -> tuple[go.Figure, str]:
    a, b = _fixed_season_limits(sub)
    if not a and not b:
        return fig, "No data for fixed-season limits."

    def add_segment(seg: SeasonSegment, color: str):
        # Draw as two blocks if it wraps (12..12 and 1..6).
        if seg.start_month == 12 and seg.end_month == 5:
            blocks = [[12], [1, 2, 3, 4, 5]]
        else:
            blocks = [list(range(seg.start_month, seg.end_month + 1))]

        for bi, x in enumerate(blocks):
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=[seg.lower] * len(x),
                    mode="lines",
                    line=dict(color=color, width=2, dash="dash"),
                    name=f"{seg.start_month}–{seg.end_month} lower",
                    legendgroup=f"fixed{seg.start_month}-{seg.end_month}",
                    showlegend=bi == 0,
                    hovertemplate=f"Fixed season lower<br>months {seg.start_month}–{seg.end_month}<br>y={seg.lower:.4g}<extra></extra>",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=[seg.upper] * len(x),
                    mode="lines",
                    line=dict(color=color, width=2, dash="dash"),
                    name=f"{seg.start_month}–{seg.end_month} upper",
                    legendgroup=f"fixed{seg.start_month}-{seg.end_month}",
                    showlegend=False,
                    hovertemplate=f"Fixed season upper<br>months {seg.start_month}–{seg.end_month}<br>y={seg.upper:.4g}<extra></extra>",
                )
            )

    if a:
        add_segment(a, "#7a1fa2")
    if b:
        add_segment(b, "#1f9e89")

    parts: list[str] = []
    if a:
        parts.append(f"12–5: lower={a.lower:.6g}, upper={a.upper:.6g}")
    else:
        parts.append("12–5: (no data)")
    if b:
        parts.append(f"6–11: lower={b.lower:.6g}, upper={b.upper:.6g}")
    else:
        parts.append("6–11: (no data)")
    return fig, "Fixed-season limits: " + " | ".join(parts)


app = Dash(__name__)
app.title = "CTD stats (Dash)"
server = app.server

init_stats_file = DEFAULT_STATS_FILE
init_df = _get_stats_df(init_stats_file)
init_variables, init_station_types, init_months = _stats_meta(init_df)

init_var = init_variables[0] if init_variables else None
init_station = init_station_types[0] if init_station_types else None
init_month = init_months[0] if init_months else None

init_bins = _depth_bin_options(init_df, init_var, init_station) if init_var and init_station else []
init_depth = init_bins[0].value if init_bins else None

if init_var and init_station and init_month is not None:
    init_fig1, init_status1 = _fig_depth_on_x(init_df, init_var, init_station, int(init_month))
else:
    init_fig1, init_status1 = go.Figure(), "No data loaded."

init_depth_options = [{"label": b.label, "value": b.value} for b in init_bins]
if init_var and init_station and init_depth:
    init_fig2, init_status2 = _fig_month_on_x(
        init_df, init_months, init_var, init_station, DepthBin.from_value(init_depth)
    )
else:
    init_fig2, init_status2 = go.Figure(), ""

if init_var and init_station and init_depth:
    _sub_init3 = _subset_month_on_x(init_df, init_var, init_station, DepthBin.from_value(init_depth))
    init_fig3, init_status3 = _add_fixed_season_limits(init_fig2, _sub_init3)
else:
    init_fig3, init_status3 = go.Figure(), ""

init_map_fig, init_map_status = _fig_station_map()

app.layout = html.Div(
    style={"maxWidth": "1100px", "margin": "0 auto", "padding": "18px 14px", "fontFamily": "system-ui"},
    children=[
        html.H2("Interactive CTD depth-bin boxplots (Dash)"),
        html.Div(
            [
                html.Div(
                    style={"display": "grid", "gridTemplateColumns": "1fr 2fr", "gap": "10px", "alignItems": "end"},
                    children=[
                        html.Div(
                            [
                                html.Div("stats file", style={"fontWeight": 600}),
                                dcc.Dropdown(
                                    id="stats-file",
                                    options=[{"label": p.name, "value": p.name} for p in STATS_PATHS],
                                    value=init_stats_file,
                                    clearable=False,
                                ),
                            ]
                        ),
                        html.Div(["Using: ", html.Code(init_stats_file)], id="stats-file-label"),
                    ],
                ),
                html.Div(
                    [
                        "Expected columns: ",
                        html.Code(", ".join(sorted(REQUIRED_COLS))),
                    ],
                    style={"color": "#555", "fontSize": "0.95rem"},
                ),
            ],
            style={"marginBottom": "10px"},
        ),
        dcc.Tabs(
            value="tab-depth",
            children=[
                dcc.Tab(
                    label="Map",
                    value="tab-map",
                    children=[
                        html.Div(
                            id="map-status",
                            children=init_map_status,
                            style={"marginTop": "10px", "fontWeight": 600, "color": "#8a2b2b"},
                        ),
                        dcc.Graph(id="map-fig", figure=init_map_fig, style={"marginTop": "6px"}),
                    ],
                ),
                dcc.Tab(
                    label="Depth bins on x (choose month)",
                    value="tab-depth",
                    children=[
                        html.Div(
                            style={
                                "display": "grid",
                                "gridTemplateColumns": "2fr 2fr 1fr",
                                "gap": "10px",
                                "alignItems": "end",
                                "marginTop": "10px",
                            },
                            children=[
                                html.Div(
                                    [
                                        html.Div("variable", style={"fontWeight": 600}),
                                        dcc.Dropdown(
                                            id="var1",
                                            options=[{"label": v, "value": v} for v in init_variables],
                                            value=init_var,
                                            clearable=False,
                                        ),
                                    ]
                                ),
                                html.Div(
                                    [
                                        html.Div("station_type", style={"fontWeight": 600}),
                                        dcc.Dropdown(
                                            id="station1",
                                            options=[{"label": s, "value": s} for s in init_station_types],
                                            value=init_station,
                                            clearable=False,
                                        ),
                                    ]
                                ),
                                html.Div(
                                    [
                                        html.Div("month", style={"fontWeight": 600}),
                                        dcc.Dropdown(
                                            id="month1",
                                            options=[{"label": str(m), "value": int(m)} for m in init_months],
                                            value=init_month,
                                            clearable=False,
                                        ),
                                    ]
                                ),
                            ],
                        ),
                        html.Div(
                            id="status1",
                            children=init_status1,
                            style={"marginTop": "8px", "fontWeight": 600, "color": "#8a2b2b"},
                        ),
                        dcc.Graph(id="fig1", figure=init_fig1, style={"marginTop": "6px"}),
                    ],
                ),
                dcc.Tab(
                    label="Months on x (choose depth bin)",
                    value="tab-month",
                    children=[
                        html.Div(
                            style={
                                "display": "grid",
                                "gridTemplateColumns": "2fr 2fr 2fr 1fr",
                                "gap": "10px",
                                "alignItems": "end",
                                "marginTop": "10px",
                            },
                            children=[
                                html.Div(
                                    [
                                        html.Div("variable", style={"fontWeight": 600}),
                                        dcc.Dropdown(
                                            id="var2",
                                            options=[{"label": v, "value": v} for v in init_variables],
                                            value=init_var,
                                            clearable=False,
                                        ),
                                    ]
                                ),
                                html.Div(
                                    [
                                        html.Div("station_type", style={"fontWeight": 600}),
                                        dcc.Dropdown(
                                            id="station2",
                                            options=[{"label": s, "value": s} for s in init_station_types],
                                            value=init_station,
                                            clearable=False,
                                        ),
                                    ]
                                ),
                                html.Div(
                                    [
                                        html.Div("depth_bin", style={"fontWeight": 600}),
                                        dcc.Dropdown(
                                            id="depth2",
                                            options=init_depth_options,
                                            value=init_depth,
                                            clearable=False,
                                        ),
                                    ]
                                ),
                                html.Div(
                                    [
                                        html.Div("seasons", style={"fontWeight": 600}),
                                        dcc.Dropdown(
                                            id="seasons2",
                                            options=[{"label": str(i), "value": i} for i in range(1, 5)],
                                            value=1,
                                            clearable=False,
                                        ),
                                    ]
                                ),
                            ],
                        ),
                        html.Div(
                            id="status2",
                            children=init_status2,
                            style={"marginTop": "8px", "fontWeight": 600, "color": "#8a2b2b"},
                        ),
                        dcc.Graph(id="fig2", figure=init_fig2, style={"marginTop": "6px"}),
                    ],
                ),
                dcc.Tab(
                    label="Months on x (fixed seasons 12–6 / 7–11)",
                    value="tab-fixed",
                    children=[
                        html.Div(
                            style={
                                "display": "grid",
                                "gridTemplateColumns": "2fr 2fr 2fr",
                                "gap": "10px",
                                "alignItems": "end",
                                "marginTop": "10px",
                            },
                            children=[
                                html.Div(
                                    [
                                        html.Div("variable", style={"fontWeight": 600}),
                                        dcc.Dropdown(
                                            id="var3",
                                            options=[{"label": v, "value": v} for v in init_variables],
                                            value=init_var,
                                            clearable=False,
                                        ),
                                    ]
                                ),
                                html.Div(
                                    [
                                        html.Div("station_type", style={"fontWeight": 600}),
                                        dcc.Dropdown(
                                            id="station3",
                                            options=[{"label": s, "value": s} for s in init_station_types],
                                            value=init_station,
                                            clearable=False,
                                        ),
                                    ]
                                ),
                                html.Div(
                                    [
                                        html.Div("depth_bin", style={"fontWeight": 600}),
                                        dcc.Dropdown(
                                            id="depth3",
                                            options=init_depth_options,
                                            value=init_depth,
                                            clearable=False,
                                        ),
                                    ]
                                ),
                            ],
                        ),
                        html.Div(
                            id="status3",
                            children=init_status3,
                            style={"marginTop": "8px", "fontWeight": 600, "color": "#8a2b2b"},
                        ),
                        dcc.Graph(id="fig3", figure=init_fig3, style={"marginTop": "6px"}),
                    ],
                ),
            ],
        ),
    ],
)

@app.callback(Output("stats-file-label", "children"), Input("stats-file", "value"))
def _update_stats_file_label(stats_file: str):
    return ["Using: ", html.Code(str(stats_file))]


@app.callback(
    Output("var1", "options"),
    Output("var1", "value"),
    Output("station1", "options"),
    Output("station1", "value"),
    Output("month1", "options"),
    Output("month1", "value"),
    Output("var2", "options"),
    Output("var2", "value"),
    Output("station2", "options"),
    Output("station2", "value"),
    Output("var3", "options"),
    Output("var3", "value"),
    Output("station3", "options"),
    Output("station3", "value"),
    Input("stats-file", "value"),
    State("var1", "value"),
    State("station1", "value"),
    State("month1", "value"),
    State("var2", "value"),
    State("station2", "value"),
    State("var3", "value"),
    State("station3", "value"),
)
def _sync_dropdown_options_for_stats_file(
    stats_file: str,
    var1: str | None,
    station1: str | None,
    month1: int | None,
    var2: str | None,
    station2: str | None,
    var3: str | None,
    station3: str | None,
):
    df = _get_stats_df(stats_file)
    variables, station_types, months = _stats_meta(df)

    def pick(current, options):
        return current if current in options else (options[0] if options else None)

    v1 = pick(var1, variables)
    s1 = pick(station1, station_types)
    m1 = pick(int(month1) if month1 is not None else None, months)
    v2 = pick(var2, variables)
    s2 = pick(station2, station_types)
    v3 = pick(var3, variables)
    s3 = pick(station3, station_types)

    return (
        [{"label": v, "value": v} for v in variables],
        v1,
        [{"label": s, "value": s} for s in station_types],
        s1,
        [{"label": str(m), "value": int(m)} for m in months],
        m1,
        [{"label": v, "value": v} for v in variables],
        v2,
        [{"label": s, "value": s} for s in station_types],
        s2,
        [{"label": v, "value": v} for v in variables],
        v3,
        [{"label": s, "value": s} for s in station_types],
        s3,
    )


@app.callback(
    Output("fig1", "figure"),
    Output("status1", "children"),
    Input("stats-file", "value"),
    Input("var1", "value"),
    Input("station1", "value"),
    Input("month1", "value"),
)
def _update_fig1(stats_file: str, var: str, station_type: str, month: int):
    df = _get_stats_df(stats_file)
    fig, status = _fig_depth_on_x(df, var, station_type, int(month))
    return fig, status


@app.callback(
    Output("depth2", "options"),
    Output("depth2", "value"),
    Output("fig2", "figure"),
    Output("status2", "children"),
    Input("stats-file", "value"),
    Input("var2", "value"),
    Input("station2", "value"),
    Input("depth2", "value"),
    Input("seasons2", "value"),
)
def _update_fig2(stats_file: str, var: str, station_type: str, depth_value: str | None, seasons: int):
    df = _get_stats_df(stats_file)
    _, _, months = _stats_meta(df)
    bins = _depth_bin_options(df, var, station_type)
    options = [{"label": b.label, "value": b.value} for b in bins]

    if not bins:
        fig = go.Figure()
        fig.update_layout(
            title=f"{var} | {station_type} (no depth bins)",
            template="plotly_white",
            height=600,
            margin=dict(l=60, r=30, t=60, b=60),
        )
        return options, None, fig, "No depth bins for this selection."

    valid_values = {b.value for b in bins}
    if depth_value not in valid_values:
        depth_value = bins[0].value

    depth_bin = DepthBin.from_value(depth_value)
    fig, status = _fig_month_on_x(df, months, var, station_type, depth_bin)

    # Add seasonal lower/upper limits that cover q1/q3 (p01/p99) and minimize slack.
    sub = _subset_month_on_x(df, var, station_type, depth_bin)
    fig2, season_status = _add_season_limits(fig, sub, seasons=seasons or 1)
    combined_status = season_status or status
    if status and season_status:
        combined_status = f"{status} | {season_status}"
    return options, depth_value, fig2, combined_status


@app.callback(
    Output("depth3", "options"),
    Output("depth3", "value"),
    Output("fig3", "figure"),
    Output("status3", "children"),
    Input("stats-file", "value"),
    Input("var3", "value"),
    Input("station3", "value"),
    Input("depth3", "value"),
)
def _update_fig3(stats_file: str, var: str, station_type: str, depth_value: str | None):
    df = _get_stats_df(stats_file)
    _, _, months = _stats_meta(df)
    bins = _depth_bin_options(df, var, station_type)
    options = [{"label": b.label, "value": b.value} for b in bins]

    if not bins:
        fig = go.Figure()
        fig.update_layout(
            title=f"{var} | {station_type} (no depth bins)",
            template="plotly_white",
            height=600,
            margin=dict(l=60, r=30, t=60, b=60),
        )
        return options, None, fig, "No depth bins for this selection."

    valid_values = {b.value for b in bins}
    if depth_value not in valid_values:
        depth_value = bins[0].value

    depth_bin = DepthBin.from_value(depth_value)
    fig, status = _fig_month_on_x(df, months, var, station_type, depth_bin)
    sub = _subset_month_on_x(df, var, station_type, depth_bin)
    fig3, limits_status = _add_fixed_season_limits(fig, sub)
    combined_status = limits_status or status
    if status and limits_status:
        combined_status = f"{status} | {limits_status}"
    return options, depth_value, fig3, combined_status


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
    #    app.run(debug=True)
