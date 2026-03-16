from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLS = {
    "variable",
    "station_type",
    "month",
    "depth_bin_low",
    "depth_bin_high",
    "n",
    "p01",
    "p99",
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
    for c in ["p01", "p99", "n"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["variable", "station_type", "month", "depth_bin_low", "depth_bin_high", "p01", "p99"])
    df["month"] = df["month"].astype(int)
    return df


def _months_str(s: pd.Series) -> str:
    # Keep it CSV-friendly but still informative.
    ms = sorted({int(v) for v in s.dropna().unique().tolist()})
    return ",".join(map(str, ms))


def compute_fixed_season_limits(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["variable", "station_type", "depth_bin_low", "depth_bin_high"]

    season_a = {12, 1, 2, 3, 4, 5}  # 12–5 (wrap)
    season_b = {6, 7, 8, 9, 10, 11}  # 6–11

    df_a = df[df["month"].isin(season_a)]
    df_b = df[df["month"].isin(season_b)]

    a = (
        df_a.groupby(group_cols, dropna=False)
        .agg(
            lower_12_5=("p01", "min"),
            upper_12_5=("p99", "max"),
            months_present_12_5=("month", _months_str),
            n_months_12_5=("month", "nunique"),
            n_sum_12_5=("n", "sum"),
        )
        .reset_index()
    )

    b = (
        df_b.groupby(group_cols, dropna=False)
        .agg(
            lower_6_11=("p01", "min"),
            upper_6_11=("p99", "max"),
            months_present_6_11=("month", _months_str),
            n_months_6_11=("month", "nunique"),
            n_sum_6_11=("n", "sum"),
        )
        .reset_index()
    )

    out = a.merge(b, on=group_cols, how="outer")
    out = out.sort_values(group_cols).reset_index(drop=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Export fixed-season limits (12–5 and 6–11) for all groups.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).resolve().parent / "stats_by_depth_all.csv",
        help="Input CSV path (default: stats_by_depth_all.csv in this folder).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "fixed_season_limit_suggestions.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    df = _load_df(args.csv)
    out = compute_fixed_season_limits(df)
    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out):,} rows to: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

