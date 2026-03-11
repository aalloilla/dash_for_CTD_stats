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

## CSV format

The Dash app uses the CSV **as-is**. It expects these columns:

`depth_bin_high`, `depth_bin_low`, `max`, `median`, `min`, `month`, `n`, `p01`, `p99`, `station_type`, `variable`

# dash_for_CTD_stats
