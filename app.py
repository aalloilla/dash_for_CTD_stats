from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html


HERE = Path(__file__).resolve().parent


def _find_default_csv() -> Path:
    csvs = sorted(HERE.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No .csv files found in {HERE}")
    return csvs[0]


CSV_PATH = _find_default_csv()


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


DF = _load_df(CSV_PATH)

VARIABLES = sorted(DF["variable"].unique().tolist())
STATION_TYPES = sorted(DF["station_type"].unique().tolist())
MONTHS = sorted(DF["month"].dropna().unique().astype(int).tolist())


def _subset_depth_on_x(var: str, station_type: str, month: int) -> pd.DataFrame:
    sub = DF[(DF["variable"] == var) & (DF["station_type"] == station_type) & (DF["month"] == month)].copy()
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


def _fig_depth_on_x(var: str, station_type: str, month: int) -> tuple[go.Figure, str]:
    sub = _subset_depth_on_x(var, station_type, month)
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


def _depth_bin_options(var: str, station_type: str) -> list[DepthBin]:
    bins = (
        DF[(DF["variable"] == var) & (DF["station_type"] == station_type)][["depth_bin_low", "depth_bin_high"]]
        .drop_duplicates()
        .dropna()
        .sort_values(["depth_bin_low", "depth_bin_high"])
    )
    out: list[DepthBin] = []
    for low, high in bins.itertuples(index=False, name=None):
        out.append(DepthBin(float(low), float(high)))
    return out


def _subset_month_on_x(var: str, station_type: str, depth_bin: DepthBin) -> pd.DataFrame:
    sub = DF[
        (DF["variable"] == var)
        & (DF["station_type"] == station_type)
        & (DF["depth_bin_low"] == depth_bin.low)
        & (DF["depth_bin_high"] == depth_bin.high)
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


def _fig_month_on_x(var: str, station_type: str, depth_bin: DepthBin) -> tuple[go.Figure, str]:
    sub = _subset_month_on_x(var, station_type, depth_bin)
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
        xaxis=dict(type="category", categoryorder="array", categoryarray=MONTHS),
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

init_var = VARIABLES[0] if VARIABLES else None
init_station = STATION_TYPES[0] if STATION_TYPES else None
init_month = MONTHS[0] if MONTHS else None

init_bins = _depth_bin_options(init_var, init_station) if init_var and init_station else []
init_depth = init_bins[0].value if init_bins else None

if init_var and init_station and init_month is not None:
    init_fig1, init_status1 = _fig_depth_on_x(init_var, init_station, int(init_month))
else:
    init_fig1, init_status1 = go.Figure(), "No data loaded."

init_depth_options = [{"label": b.label, "value": b.value} for b in init_bins]
if init_var and init_station and init_depth:
    init_fig2, init_status2 = _fig_month_on_x(init_var, init_station, DepthBin.from_value(init_depth))
else:
    init_fig2, init_status2 = go.Figure(), ""

if init_var and init_station and init_depth:
    _sub_init3 = _subset_month_on_x(init_var, init_station, DepthBin.from_value(init_depth))
    init_fig3, init_status3 = _add_fixed_season_limits(init_fig2, _sub_init3)
else:
    init_fig3, init_status3 = go.Figure(), ""

app.layout = html.Div(
    style={"maxWidth": "1100px", "margin": "0 auto", "padding": "18px 14px", "fontFamily": "system-ui"},
    children=[
        html.H2("Interactive CTD depth-bin boxplots (Dash)"),
        html.Div(
            [
                html.Div(["CSV: ", html.Code(str(CSV_PATH.name))]),
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
                                            options=[{"label": v, "value": v} for v in VARIABLES],
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
                                            options=[{"label": s, "value": s} for s in STATION_TYPES],
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
                                            options=[{"label": str(m), "value": int(m)} for m in MONTHS],
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
                                            options=[{"label": v, "value": v} for v in VARIABLES],
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
                                            options=[{"label": s, "value": s} for s in STATION_TYPES],
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
                                            options=[{"label": v, "value": v} for v in VARIABLES],
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
                                            options=[{"label": s, "value": s} for s in STATION_TYPES],
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


@app.callback(Output("fig1", "figure"), Output("status1", "children"), Input("var1", "value"), Input("station1", "value"), Input("month1", "value"))
def _update_fig1(var: str, station_type: str, month: int):
    fig, status = _fig_depth_on_x(var, station_type, int(month))
    return fig, status


@app.callback(
    Output("depth2", "options"),
    Output("depth2", "value"),
    Output("fig2", "figure"),
    Output("status2", "children"),
    Input("var2", "value"),
    Input("station2", "value"),
    Input("depth2", "value"),
    Input("seasons2", "value"),
)
def _update_fig2(var: str, station_type: str, depth_value: str | None, seasons: int):
    bins = _depth_bin_options(var, station_type)
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
    fig, status = _fig_month_on_x(var, station_type, depth_bin)

    # Add seasonal lower/upper limits that cover q1/q3 (p01/p99) and minimize slack.
    sub = _subset_month_on_x(var, station_type, depth_bin)
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
    Input("var3", "value"),
    Input("station3", "value"),
    Input("depth3", "value"),
)
def _update_fig3(var: str, station_type: str, depth_value: str | None):
    bins = _depth_bin_options(var, station_type)
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
    fig, status = _fig_month_on_x(var, station_type, depth_bin)
    sub = _subset_month_on_x(var, station_type, depth_bin)
    fig3, limits_status = _add_fixed_season_limits(fig, sub)
    combined_status = limits_status or status
    if status and limits_status:
        combined_status = f"{status} | {limits_status}"
    return options, depth_value, fig3, combined_status


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
    #    app.run(debug=True)
