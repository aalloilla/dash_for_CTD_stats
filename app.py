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
)
def _update_fig2(var: str, station_type: str, depth_value: str | None):
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
    return options, depth_value, fig, status


if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8080)
    #    app.run(debug=True)
