# CTD stats Dash app

This folder contains:

- `stats_by_depth_all.csv`: summary stats by variable / station_type / month / depth-bin
- `app.py`: a Dash app that recreates the notebook’s interactive Plotly views

## Run

```bash
python -m pip install -r requirements.txt
python app.py
```

Then open the URL printed in the terminal (typically `http://127.0.0.1:8050`).

## Map tab notes

The map tab uses `shapely` to determine whether each station point is inside the shallow polygon.

## Export fixed-season limit suggestions (all combinations)

This generates limit suggestions for **every** `(variable, station_type, depth_bin_low, depth_bin_high)` using the same fixed seasons as the 3rd plot:

- **Season A**: months **12–5** (wrap) = 12,1,2,3,4,5
- **Season B**: months **6–11** = 6,7,8,9,10,11

```bash
python export_fixed_season_limits.py
```

Output file: `fixed_season_limit_suggestions.csv`

## CSV format

The Dash app uses the CSV **as-is**. It expects these columns:

`depth_bin_high`, `depth_bin_low`, `max`, `median`, `min`, `month`, `n`, `p01`, `p99`, `station_type`, `variable`

# dash_for_CTD_stats
