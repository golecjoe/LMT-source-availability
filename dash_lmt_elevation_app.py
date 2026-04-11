#!/usr/bin/env python3
"""Interactive Dash app for LMT source elevation plots."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_sun
from astropy.time import Time
from dash import Dash, Input, Output, dcc, html
import plotly.graph_objects as go


LMT_LOCATION = EarthLocation(lat=18.985 * u.deg, lon=-97.314 * u.deg, height=4600 * u.m)
DEFAULT_CATALOG = Path("catalogs/ALMA_calibrators.cat")
DEFAULT_TZ = "America/Mexico_City"
MIN_TRANSIT_ALT_DEG = 30.0
YMIN, YMAX = 20.0, 92.0
CATALOG_DIR = Path("catalogs")


@dataclass
class CatalogSource:
    name: str
    ra: str
    dec: str
    flux_jy: float


def parse_catalog(path: Path) -> list[CatalogSource]:
    sources: list[CatalogSource] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#") or s.startswith("NAME"):
                continue
            parts = s.split()
            if len(parts) < 7:
                continue
            sources.append(
                CatalogSource(
                    name=parts[0],
                    ra=parts[2],
                    dec=parts[3],
                    flux_jy=float(parts[6]),
                )
            )
    return sources


def catalog_options(catalog_dir: Path) -> list[dict[str, str]]:
    files = sorted(catalog_dir.glob("*.cat"))
    return [{"label": p.name, "value": str(p)} for p in files]


def upper_transit_altitude_deg(dec_deg: float, site_lat_deg: float) -> float:
    return 90.0 - abs(site_lat_deg - dec_deg)


def filter_sources(
    sources: list[CatalogSource],
    min_flux: float | None,
    max_flux: float | None,
    min_transit_alt_deg: float,
) -> list[CatalogSource]:
    site_lat_deg = float(LMT_LOCATION.lat.to_value(u.deg))
    kept: list[CatalogSource] = []
    for src in sources:
        if min_flux is not None and src.flux_jy < min_flux:
            continue
        if max_flux is not None and src.flux_jy > max_flux:
            continue
        dec_deg = float(SkyCoord(src.ra, src.dec, unit=(u.hourangle, u.deg), frame="icrs").dec.deg)
        if upper_transit_altitude_deg(dec_deg, site_lat_deg) < min_transit_alt_deg:
            continue
        kept.append(src)
    return kept


def _sun_altitudes_deg(datetimes_utc: list[datetime]) -> list[float]:
    times = Time(datetimes_utc, scale="utc")
    altaz = AltAz(obstime=times, location=LMT_LOCATION)
    return get_sun(times).transform_to(altaz).alt.deg.tolist()


def _find_altitude_crossing(
    datetimes_utc: list[datetime], altitudes_deg: list[float], rising: bool, target_deg: float = 0.0
) -> datetime | None:
    for i in range(1, len(datetimes_utc)):
        a0 = altitudes_deg[i - 1]
        a1 = altitudes_deg[i]
        t0 = datetimes_utc[i - 1]
        t1 = datetimes_utc[i + 0]
        if rising:
            crossed = a0 <= target_deg and a1 > target_deg
        else:
            crossed = a0 > target_deg and a1 <= target_deg
        if not crossed:
            continue
        if a1 == a0:
            return t0
        frac = (target_deg - a0) / (a1 - a0)
        return t0 + (t1 - t0) * frac
    return None


def sunset_sunrise_for_lmt_night(night_date_local: date, lmt_tz: ZoneInfo) -> tuple[datetime, datetime]:
    sunset_start_local = datetime.combine(night_date_local, datetime.min.time(), tzinfo=lmt_tz).replace(hour=12)
    sunset_end_local = sunset_start_local + timedelta(hours=14)

    sunrise_start_local = sunset_start_local + timedelta(hours=12)
    sunrise_end_local = sunrise_start_local + timedelta(hours=12)

    def make_utc_grid(start_local: datetime, end_local: datetime) -> list[datetime]:
        out: list[datetime] = []
        t = start_local.astimezone(timezone.utc)
        end_utc = end_local.astimezone(timezone.utc)
        step = timedelta(minutes=2)
        while t <= end_utc:
            out.append(t)
            t += step
        if out[-1] != end_utc:
            out.append(end_utc)
        return out

    sunset_grid_utc = make_utc_grid(sunset_start_local, sunset_end_local)
    sunset_utc = _find_altitude_crossing(sunset_grid_utc, _sun_altitudes_deg(sunset_grid_utc), rising=False)

    sunrise_grid_utc = make_utc_grid(sunrise_start_local, sunrise_end_local)
    sunrise_utc = _find_altitude_crossing(sunrise_grid_utc, _sun_altitudes_deg(sunrise_grid_utc), rising=True)

    if sunset_utc is None or sunrise_utc is None:
        raise RuntimeError("Could not determine sunset/sunrise for requested LMT night date.")
    return sunset_utc, sunrise_utc


def utc_grid_between(start_utc: datetime, end_utc: datetime, step_minutes: int = 10) -> tuple[list[datetime], Time]:
    out: list[datetime] = []
    t = start_utc
    step = timedelta(minutes=step_minutes)
    while t <= end_utc:
        out.append(t)
        t += step
    if out[-1] != end_utc:
        out.append(end_utc)
    return out, Time(out, scale="utc")


def build_figure(
    sources: list[CatalogSource],
    utc_datetimes: list[datetime],
    times_astropy: Time,
    sunset_utc: datetime,
    sunrise_utc: datetime,
    night_date_local: date,
    min_flux: float | None,
    max_flux: float | None,
) -> go.Figure:
    frame = AltAz(obstime=times_astropy, location=LMT_LOCATION)
    lst_deg = times_astropy.sidereal_time("apparent", longitude=LMT_LOCATION.lon).deg

    fig = go.Figure()

    label_x: list[datetime] = []
    label_y: list[float] = []
    label_text: list[str] = []

    for src in sources:
        coord = SkyCoord(src.ra, src.dec, unit=(u.hourangle, u.deg), frame="icrs")
        alt = coord.transform_to(frame).alt.deg
        fig.add_trace(
            go.Scatter(
                x=utc_datetimes,
                y=alt,
                mode="lines",
                name=src.name,
                line={"width": 1.0},
                opacity=0.25,
                customdata=[[src.name, src.flux_jy]] * len(utc_datetimes),
                hovertemplate="%{customdata[0]}<br>Flux=%{customdata[1]:.3f} Jy<br>UTC=%{x|%Y-%m-%d %H:%M}<br>El=%{y:.1f} deg<extra></extra>",
            )
        )

        ha_deg = ((lst_deg - coord.ra.deg + 540.0) % 360.0) - 180.0
        transit_time_utc: datetime | None = None
        for i in range(len(utc_datetimes) - 1):
            h0, h1 = ha_deg[i], ha_deg[i + 1]
            if h0 <= 0.0 and h1 > 0.0:
                frac = 0.0 if h1 == h0 else (0.0 - h0) / (h1 - h0)
                transit_time_utc = utc_datetimes[i] + (utc_datetimes[i + 1] - utc_datetimes[i]) * frac
                break

        if transit_time_utc is None:
            continue

        transit_alt = (
            coord.transform_to(AltAz(obstime=Time(transit_time_utc, scale="utc"), location=LMT_LOCATION))
            .alt.deg
        )
        if transit_alt >= YMIN:
            label_x.append(transit_time_utc)
            label_y.append(min(transit_alt + 1.0, 89.5))
            label_text.append(src.name)

    if label_x:
        fig.add_trace(
            go.Scatter(
                x=label_x,
                y=label_y,
                mode="text",
                text=label_text,
                textposition="top center",
                textfont={"size": 9, "color": "rgba(40,40,40,0.9)"},
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig.add_hline(y=0.0, line_dash="dash", line_color="black", line_width=1)
    fig.add_vline(x=sunset_utc, line_dash="dot", line_color="orange", line_width=1.5)
    fig.add_vline(x=sunrise_utc, line_dash="dot", line_color="deepskyblue", line_width=1.5)
    fig.add_vrect(x0=utc_datetimes[0], x1=sunset_utc, fillcolor="gray", opacity=0.2, line_width=0)
    fig.add_vrect(x0=sunrise_utc, x1=utc_datetimes[-1], fillcolor="gray", opacity=0.2, line_width=0)

    flux_desc = f"Flux min={min_flux}, max={max_flux} Jy"
    fig.update_layout(
        title=f"LMT Source Elevations ({night_date_local.isoformat()} local night, UTC axis)",
        xaxis_title="UTC Time",
        yaxis_title="Elevation (deg)",
        yaxis={"range": [YMIN, YMAX]},
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="closest",
        showlegend=False,
        margin={"l": 60, "r": 20, "t": 65, "b": 55},
        meta={"num_source_traces": len(sources)},
        annotations=[
            {
                "xref": "paper",
                "yref": "paper",
                "x": 0,
                "y": 1.08,
                "showarrow": False,
                "text": (
                    f"Transit filter >= {MIN_TRANSIT_ALT_DEG:.0f} deg, {flux_desc}, "
                    f"sources={len(sources)}"
                ),
                "font": {"size": 12, "color": "#444"},
            }
        ],
    )
    fig.update_xaxes(tickformat="%m-%d\n%H:%M")
    return fig


def default_night_date_str(lmt_tz_name: str) -> str:
    return datetime.now(ZoneInfo(lmt_tz_name)).date().isoformat()


def build_base_figure(
    catalog_path: Path,
    night_date_str: str,
    min_flux: float | None,
    max_flux: float | None,
    lmt_tz_name: str,
) -> go.Figure:
    if min_flux is not None and max_flux is not None and min_flux > max_flux:
        raise ValueError("Min flux cannot be greater than max flux.")

    lmt_tz = ZoneInfo(lmt_tz_name)
    night_date_local = datetime.strptime(night_date_str, "%Y-%m-%d").date()

    all_sources = parse_catalog(catalog_path)
    sources = filter_sources(
        all_sources,
        min_flux=min_flux,
        max_flux=max_flux,
        min_transit_alt_deg=MIN_TRANSIT_ALT_DEG,
    )

    sunset_utc, sunrise_utc = sunset_sunrise_for_lmt_night(night_date_local, lmt_tz)
    start_utc = sunset_utc - timedelta(hours=2)
    end_utc = sunrise_utc + timedelta(hours=2)
    utc_datetimes, times_astropy = utc_grid_between(start_utc, end_utc, step_minutes=10)

    return build_figure(
        sources=sources,
        utc_datetimes=utc_datetimes,
        times_astropy=times_astropy,
        sunset_utc=sunset_utc,
        sunrise_utc=sunrise_utc,
        night_date_local=night_date_local,
        min_flux=min_flux,
        max_flux=max_flux,
    )


app = Dash(__name__)
_catalog_opts = catalog_options(CATALOG_DIR)
_default_catalog_value = (
    str(DEFAULT_CATALOG)
    if any(opt["value"] == str(DEFAULT_CATALOG) for opt in _catalog_opts)
    else (_catalog_opts[0]["value"] if _catalog_opts else str(DEFAULT_CATALOG))
)

app.layout = html.Div(
    style={"fontFamily": "Arial, sans-serif", "padding": "12px"},
    children=[
        html.H3("LMT Source Elevation Planner", style={"margin": "0 0 10px 0"}),
        html.Div(
            style={"display": "flex", "gap": "10px", "alignItems": "center", "marginBottom": "10px"},
            children=[
                html.Label("Catalog:"),
                dcc.Dropdown(
                    id="catalog-path",
                    options=_catalog_opts,
                    value=_default_catalog_value,
                    clearable=False,
                    style={"minWidth": "280px"},
                ),
                html.Label("Night Date (LMT local):"),
                dcc.Input(id="night-date", type="text", value=default_night_date_str(DEFAULT_TZ), debounce=True),
                html.Label("Min Flux (Jy):"),
                dcc.Input(id="min-flux", type="number", debounce=True),
                html.Label("Max Flux (Jy):"),
                dcc.Input(id="max-flux", type="number", debounce=True),
                html.Div(id="status", style={"marginLeft": "8px", "color": "#444"}),
            ],
        ),
        dcc.Store(id="base-fig-store"),
        dcc.Graph(id="elevation-graph", clear_on_unhover=True, style={"height": "82vh"}),
    ],
)


@app.callback(
    Output("base-fig-store", "data"),
    Output("status", "children"),
    Input("catalog-path", "value"),
    Input("night-date", "value"),
    Input("min-flux", "value"),
    Input("max-flux", "value"),
)
def recompute_base_figure(
    catalog_path_value: str, night_date: str, min_flux: float | None, max_flux: float | None
):
    try:
        fig = build_base_figure(
            catalog_path=Path(catalog_path_value),
            night_date_str=night_date or default_night_date_str(DEFAULT_TZ),
            min_flux=min_flux,
            max_flux=max_flux,
            lmt_tz_name=DEFAULT_TZ,
        )
        return fig.to_dict(), ""
    except Exception as exc:
        msg = f"Input error: {exc}"
        empty = go.Figure()
        empty.update_layout(title="LMT Source Elevations", xaxis_title="UTC Time", yaxis_title="Elevation (deg)")
        return empty.to_dict(), msg


@app.callback(
    Output("elevation-graph", "figure"),
    Input("base-fig-store", "data"),
    Input("elevation-graph", "hoverData"),
)
def render_with_hover_highlight(base_fig_data, hover_data):
    fig = go.Figure(base_fig_data) if base_fig_data else go.Figure()

    num_source_traces = int((fig.layout.meta or {}).get("num_source_traces", 0))

    for i in range(num_source_traces):
        fig.data[i].line.width = 1.0
        fig.data[i].opacity = 0.25

    if hover_data and hover_data.get("points"):
        curve_idx = hover_data["points"][0].get("curveNumber")
        if isinstance(curve_idx, int) and 0 <= curve_idx < num_source_traces:
            fig.data[curve_idx].line.width = 3.0
            fig.data[curve_idx].opacity = 1.0

    return fig


if __name__ == "__main__":
    app.run(debug=True, port=8050)
