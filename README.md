# Interactive Source Availability

Tools for plotting radio-source elevation vs UTC time at the LMT site.

## Files

- `plot_lmt_source_elevations.py`: Static PNG plot generator.
- `dash_lmt_elevation_app.py`: Interactive Dash web app.
- `catalogs/*.cat`: Source catalogs (ALMA/SMA).

## Requirements

Use Python 3.10+ and install:

```bash
pip install astropy matplotlib dash plotly
```

## Run Static Plot Script

Generate a plot for tonight (LMT local night by default):

```bash
python3 plot_lmt_source_elevations.py \
  --catalog catalogs/ALMA_calibrators.cat \
  --output lmt_alma_elevation_plot_tonight.png
```

Useful options:

- `--night-date YYYY-MM-DD`: LMT local date of the observing night.
- `--min-flux FLOAT`: Minimum source flux (Jy).
- `--max-flux FLOAT`: Maximum source flux (Jy).
- `--min-transit-alt-deg FLOAT`: Transit altitude filter (default `30`).
- `--step-minutes INT`: Time sampling in minutes.
- `--output PATH`: Output PNG path.

Example:

```bash
python3 plot_lmt_source_elevations.py \
  --catalog catalogs/SMA_calibrators.cat \
  --night-date 2026-04-15 \
  --min-flux 0.2 \
  --max-flux 1.5 \
  --output sma_night_plot.png
```

## Run Dash App

Start the interactive app:

```bash
python3 dash_lmt_elevation_app.py
```

Open in browser:

- `http://127.0.0.1:8050`

In the UI, use the top controls to change:

- Catalog (from `catalogs/`)
- Night date (LMT local date)
- Min/Max flux

Hovering a trace highlights that source.

## Stopping the Dash Server

- Use `Ctrl+C` in the terminal running the app.
- Do not use `Ctrl+Z` (that suspends the process and can leave port `8050` in use).
