#!/usr/bin/env python3
"""Plot source elevation versus UTC time for LMT using a radio-source catalog."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_sun
from astropy.time import Time


# Large Millimeter Telescope Alfonso Serrano (Sierra Negra, Mexico)
LMT_LOCATION = EarthLocation(lat=18.985 * u.deg, lon=-97.314 * u.deg, height=4600 * u.m)


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
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("NAME"):
                continue

            parts = stripped.split()
            if len(parts) < 7:
                continue

            name = parts[0]
            ra = parts[2]
            dec = parts[3]
            flux_jy = float(parts[6])
            sources.append(CatalogSource(name=name, ra=ra, dec=dec, flux_jy=flux_jy))
    return sources


def upper_transit_altitude_deg(dec_deg: float, site_lat_deg: float) -> float:
    return 90.0 - abs(site_lat_deg - dec_deg)


def filter_sources_by_transit_altitude(
    sources: list[CatalogSource], min_transit_alt_deg: float
) -> list[CatalogSource]:
    site_lat_deg = float(LMT_LOCATION.lat.to_value(u.deg))
    kept: list[CatalogSource] = []
    for src in sources:
        dec_deg = float(SkyCoord(src.ra, src.dec, unit=(u.hourangle, u.deg), frame="icrs").dec.deg)
        if upper_transit_altitude_deg(dec_deg, site_lat_deg) >= min_transit_alt_deg:
            kept.append(src)
    return kept


def filter_sources_by_flux(
    sources: list[CatalogSource], min_flux: float | None, max_flux: float | None
) -> list[CatalogSource]:
    kept: list[CatalogSource] = []
    for src in sources:
        if min_flux is not None and src.flux_jy < min_flux:
            continue
        if max_flux is not None and src.flux_jy > max_flux:
            continue
        kept.append(src)
    return kept


def utc_grid_between(
    start_utc: datetime, end_utc: datetime, step_minutes: int
) -> tuple[list[datetime], Time]:
    if end_utc <= start_utc:
        raise ValueError("end_utc must be greater than start_utc")

    utc_datetimes: list[datetime] = []
    t = start_utc
    step = timedelta(minutes=step_minutes)
    while t <= end_utc:
        utc_datetimes.append(t)
        t += step
    if utc_datetimes[-1] != end_utc:
        utc_datetimes.append(end_utc)

    return utc_datetimes, Time(utc_datetimes, scale="utc")


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
        t1 = datetimes_utc[i]

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


def sunset_sunrise_for_lmt_night(
    night_date_local: date, lmt_tz: ZoneInfo
) -> tuple[datetime, datetime]:
    # "night_date_local" means the local date of sunset at LMT.
    sunset_start_local = datetime.combine(night_date_local, datetime.min.time(), tzinfo=lmt_tz).replace(
        hour=12
    )
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
    sunset_alts = _sun_altitudes_deg(sunset_grid_utc)
    sunset_utc = _find_altitude_crossing(sunset_grid_utc, sunset_alts, rising=False)

    sunrise_grid_utc = make_utc_grid(sunrise_start_local, sunrise_end_local)
    sunrise_alts = _sun_altitudes_deg(sunrise_grid_utc)
    sunrise_utc = _find_altitude_crossing(sunrise_grid_utc, sunrise_alts, rising=True)

    if sunset_utc is None or sunrise_utc is None:
        raise RuntimeError("Could not determine sunset/sunrise for requested LMT night date.")
    return sunset_utc, sunrise_utc


def make_plot(
    sources: list[CatalogSource],
    utc_datetimes: list[datetime],
    times_astropy: Time,
    output: Path,
    night_date_local: date,
    sunset_utc: datetime,
    sunrise_utc: datetime,
) -> None:
    frame = AltAz(obstime=times_astropy, location=LMT_LOCATION)
    lst_deg = times_astropy.sidereal_time("apparent", longitude=LMT_LOCATION.lon).deg

    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)

    for src in sources:
        coord = SkyCoord(src.ra, src.dec, unit=(u.hourangle, u.deg), frame="icrs")
        alt = coord.transform_to(frame).alt.deg
        ax.plot(utc_datetimes, alt, linewidth=0.8, alpha=0.28)

        # Label each source at its meridian transit if transit occurs in plotted window.
        ha_deg = ((lst_deg - coord.ra.deg + 540.0) % 360.0) - 180.0
        transit_time_utc: datetime | None = None
        for i in range(len(utc_datetimes) - 1):
            h0 = ha_deg[i]
            h1 = ha_deg[i + 1]
            if h0 <= 0.0 and h1 > 0.0:
                frac = 0.0 if h1 == h0 else (0.0 - h0) / (h1 - h0)
                transit_time_utc = utc_datetimes[i] + (utc_datetimes[i + 1] - utc_datetimes[i]) * frac
                break
        if transit_time_utc is None:
            continue

        transit_alt = (
            coord.transform_to(
                AltAz(obstime=Time(transit_time_utc, scale="utc"), location=LMT_LOCATION)
            )
            .alt.deg
        )
        ax.text(
            transit_time_utc,
            min(transit_alt + 1.0, 89.5),
            src.name,
            fontsize=6,
            alpha=0.8,
            ha="center",
            va="bottom",
            clip_on=True,
        )

    plot_start_utc = utc_datetimes[0]
    plot_end_utc = utc_datetimes[-1]
    ax.axvspan(
        plot_start_utc,
        sunset_utc,
        color="gray",
        alpha=0.2,
        label="Before sunset / after sunrise",
    )
    ax.axvspan(
        sunrise_utc,
        plot_end_utc,
        color="gray",
        alpha=0.2,
    )

    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, label="Horizon")
    ax.axvline(sunset_utc, color="tab:orange", linestyle=":", linewidth=1.2, label="Sunset (UTC)")
    ax.axvline(sunrise_utc, color="tab:cyan", linestyle=":", linewidth=1.2, label="Sunrise (UTC)")
    ax.set_ylim(20, 92)
    ax.set_ylabel("Elevation (deg)")
    ax.set_xlabel("UTC Time")
    ax.set_title(
        f"LMT Source Elevations ({night_date_local.isoformat()} local night, UTC axis)"
    )

    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2, tz=timezone.utc))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M", tz=timezone.utc))

    ax.grid(True, alpha=0.35)
    ax.legend(loc="upper right")

    fig.savefig(output, dpi=180)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot source elevation vs UTC time for LMT from a radio-source catalog."
        )
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("catalogs/ALMA_calibrators.cat"),
        help="Path to source catalog file.",
    )
    parser.add_argument(
        "--night-date",
        type=str,
        default=None,
        help=(
            "LMT local date of the observing night (YYYY-MM-DD). "
            "Default: tonight at LMT local timezone."
        ),
    )
    parser.add_argument(
        "--lmt-timezone",
        type=str,
        default="America/Mexico_City",
        help="IANA timezone used for 'tonight' and night-date interpretation.",
    )
    parser.add_argument(
        "--step-minutes",
        type=int,
        default=10,
        help="Sampling interval in minutes (default: 10).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("lmt_alma_elevation_plot.png"),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--max-sources",
        type=int,
        default=None,
        help=(
            "If set, keep only this many brightest sources (by Flux column) to reduce clutter."
        ),
    )
    parser.add_argument(
        "--min-transit-alt-deg",
        type=float,
        default=30.0,
        help="Only plot sources whose upper transit altitude at LMT is at least this value (deg).",
    )
    parser.add_argument(
        "--min-flux",
        type=float,
        default=None,
        help="Minimum catalog flux (Jy) to include.",
    )
    parser.add_argument(
        "--max-flux",
        type=float,
        default=None,
        help="Maximum catalog flux (Jy) to include.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.step_minutes <= 0:
        raise ValueError("--step-minutes must be > 0")
    if args.min_flux is not None and args.max_flux is not None and args.min_flux > args.max_flux:
        raise ValueError("--min-flux cannot be greater than --max-flux")

    lmt_tz = ZoneInfo(args.lmt_timezone)
    if args.night_date is None:
        night_date_local = datetime.now(lmt_tz).date()
    else:
        night_date_local = datetime.strptime(args.night_date, "%Y-%m-%d").date()

    sources = parse_catalog(args.catalog)
    if not sources:
        raise RuntimeError(f"No sources parsed from catalog: {args.catalog}")

    sources = filter_sources_by_flux(
        sources, min_flux=args.min_flux, max_flux=args.max_flux
    )

    if args.max_sources is not None:
        if args.max_sources <= 0:
            raise ValueError("--max-sources must be > 0")
        sources = sorted(sources, key=lambda s: s.flux_jy, reverse=True)[: args.max_sources]

    sources = filter_sources_by_transit_altitude(
        sources, min_transit_alt_deg=args.min_transit_alt_deg
    )
    if not sources:
        raise RuntimeError(
            "No sources remain after filtering. "
            "Try adjusting --min-flux/--max-flux and/or --min-transit-alt-deg."
        )

    sunset_utc, sunrise_utc = sunset_sunrise_for_lmt_night(night_date_local, lmt_tz)
    plot_start_utc = sunset_utc - timedelta(hours=2)
    plot_end_utc = sunrise_utc + timedelta(hours=2)

    utc_datetimes, times_astropy = utc_grid_between(
        plot_start_utc, plot_end_utc, args.step_minutes
    )
    make_plot(
        sources=sources,
        utc_datetimes=utc_datetimes,
        times_astropy=times_astropy,
        output=args.output,
        night_date_local=night_date_local,
        sunset_utc=sunset_utc,
        sunrise_utc=sunrise_utc,
    )

    print(
        f"Saved plot to {args.output} using {len(sources)} sources sampled every "
        f"{args.step_minutes} min.\n"
        f"Flux filter (Jy): min={args.min_flux} max={args.max_flux}\n"
        f"Min transit altitude filter: {args.min_transit_alt_deg:.1f} deg\n"
        f"LMT night date: {night_date_local.isoformat()} ({args.lmt_timezone})\n"
        f"Sunset UTC: {sunset_utc.isoformat()}\n"
        f"Sunrise UTC: {sunrise_utc.isoformat()}\n"
        f"Window UTC: {plot_start_utc.isoformat()} to {plot_end_utc.isoformat()}"
    )


if __name__ == "__main__":
    main()
