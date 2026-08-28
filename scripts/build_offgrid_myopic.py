# SPDX-FileCopyrightText: 2026 Mohamed Amine Chebaane

# SPDX-License-Identifier: CC-BY-4.0
"""
build_offgrid_myopic.py
=======================
Bachelorarbeit: Offgrid-Integration in PyPSA-Earth
OTH Regensburg | Betreuer: Anton Achhammer, Prof. Michael Sterner

PyPSA-Earth Skript: Identifiziert Offgrid-Regionen und fuegt isolierte
Mini-Grid Busse + Komponenten ins Netzwerk ein.

Generisch – funktioniert fuer jedes Land in config["countries"].
Elektrifizierungsdaten: {COUNTRY}_electricity_access.csv pro Land.

Ablauf:
    1. Laedt geloestes PyPSA-Earth Netzwerk
    2. Liest Offgrid-Parameter aus config.yaml (snakemake.config["offgrid"])
    3. Geo-Logik (C1-C5): Offgrid-Counties identifizieren
    4. Fuegt isolierte Mini-Grid Busse ins Netzwerk ein
    5. Optimiert erweitertes Netzwerk
    6. Speichert Ergebnisse

Einheiten in PyPSA (Quelle: PyPSA Dokumentation):
    - Leistung:      MW
    - Energie:       MWh
    - capital_cost:  EUR/MW/a  (annualisiert)
    - marginal_cost: EUR/MWh

Quellen:
    - Osiolo et al. (2019): 60 kWh/yr rural Kenya
    - PyPSA technology-data (TU Berlin/DEA): Solar/Batterie CAPEX
    - ERA5: Hersbach et al. (2020), doi:10.1002/qj.3803
    - ESMAP Mini Grid Design Manual (2019): Annuität r=8%, n=20yr
    - IEA Africa Energy Outlook (2022): Netzanschlusskosten
"""

import logging
import os
import warnings

import atlite
import geopandas as gpd
import numpy as np
import pandas as pd
import pypsa
from shapely.geometry import Point

try:
    from offgrid_costs import build_offgrid_costs
except ModuleNotFoundError:  # pragma: no cover - direct import from repo root
    from scripts.offgrid_costs import build_offgrid_costs

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════
# HILFSFUNKTIONEN – Elektrifizierungsdaten
# ══════════════════════════════════════════════════════


def load_electrification_data_from_path(path):
    """
    Laedt Elektrifizierungsdaten aus einer CSV-Datei.

    Generisch – Dateiname bestimmt Land (z.B. KE_electricity_access.csv).
    Format:
        GADM_ID,                Access to electricity
        National Average,       23.0
        KE.1_1,                 9.6

    Returns:
        elec_dict (dict): {GADM_ID: rate}
        national_avg (float): nationaler Durchschnitt
    """
    if not os.path.exists(path):
        logger.warning(f"Elektrifizierungsdaten nicht gefunden: {path}")
        return None, 0.23

    df = pd.read_csv(path)
    nat_row = df[df["GADM_ID"] == "National Average"]["Access to electricity"].values
    nat_avg = float(nat_row[0]) / 100 if len(nat_row) > 0 else 0.23
    df_reg = df[df["GADM_ID"] != "National Average"].dropna(subset=["GADM_ID"])
    elec_dict = dict(
        zip(
            df_reg["GADM_ID"].str.strip(),
            df_reg["Access to electricity"].astype(float) / 100,
        )
    )
    country = os.path.basename(path).split("_")[0]
    logger.info(f"  {country}: {len(elec_dict)} Regionen | Ø {nat_avg*100:.1f}%")
    return elec_dict, nat_avg


def load_all_electrification_data(elec_data_files):
    """
    Laedt Elektrifizierungsdaten fuer alle Laender.

    Iteriert ueber alle CSV-Dateien in snakemake.input.elec_data.
    Kombiniert alle Laender in einem Dictionary.

    Args:
        elec_data_files: str oder Liste von Pfaden

    Returns:
        elec_data_combined (dict): {GADM_ID: rate} fuer alle Laender
        national_avg (float): letzter nationaler Durchschnitt (Fallback)
    """
    # Snakemake gibt entweder str (1 Datei) oder Liste zurueck
    if isinstance(elec_data_files, str):
        elec_data_files = [elec_data_files]

    elec_data_combined = {}
    national_avg = 0.23  # Fallback

    for path in elec_data_files:
        ed, nav = load_electrification_data_from_path(path)
        if ed:
            elec_data_combined.update(ed)
            national_avg = nav  # letzter Wert als Fallback

    if not elec_data_combined:
        logger.warning("Keine Elektrifizierungsdaten gefunden → Heuristik")
        return None, national_avg

    logger.info(
        f"Gesamt: {len(elec_data_combined)} Regionen aus {len(elec_data_files)} Laendern"
    )
    return elec_data_combined, national_avg


def get_electrification_rate(
    gadm_id, distance_km, pop_density, elec_data=None, national_avg=0.23
):
    """
    Gibt Elektrifizierungsrate zurueck – echte Daten oder Heuristik.

    Prioritaet 1: Echte CSV-Daten
    Prioritaet 2: Heuristik nach Netzabstand + Bevoelkerungsdichte
    """
    if elec_data is not None:
        if gadm_id in elec_data:
            return elec_data[gadm_id], "Census"
        return national_avg, "National Avg"

    # Heuristik – Quelle: Weltbank (2022), eigene Ableitung
    if distance_km > 100:
        base = 0.15
    elif distance_km > 50:
        base = 0.30
    elif distance_km > 25:
        base = 0.55
    else:
        base = 0.76

    if pop_density > 500:
        f = 1.2
    elif pop_density > 100:
        f = 1.0
    elif pop_density > 25:
        f = 0.8
    else:
        f = 0.6

    return round(min(base * f, 1.0), 4), "Heuristik"


# ══════════════════════════════════════════════════════
# HILFSFUNKTIONEN – CD 2035 Standalone-Population
# ══════════════════════════════════════════════════════


def get_named_value(container, name, default=None):
    """Read a value from a Snakemake namedlist-like object or a plain mapping."""
    if hasattr(container, name):
        return getattr(container, name)
    if isinstance(container, dict):
        return container.get(name, default)
    try:
        return container[name]
    except Exception:
        return default


def load_access_region_mapping(path):
    """Load bus -> access-region mapping used by the CD demand split."""
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Access-region mapping file not found: {path}")

    mapping = pd.read_csv(path)
    region_col = "access_region" if "access_region" in mapping.columns else "region"
    required_cols = {"bus", region_col}
    if not required_cols.issubset(mapping.columns):
        raise ValueError(
            "Access-region mapping must contain bus and either region or "
            f"access_region. Found {set(mapping.columns)}"
        )

    mapping = mapping.rename(columns={region_col: "access_region"})
    mapping = mapping.assign(
        bus=lambda df: df["bus"].astype(str).str.strip(),
        access_region=lambda df: df["access_region"].astype(str).str.strip(),
    )
    mapping = mapping[mapping["bus"] != ""]
    if mapping["bus"].duplicated().any():
        duplicates = mapping.loc[mapping["bus"].duplicated(), "bus"].tolist()
        raise ValueError(
            "Access-region mapping contains duplicate buses: "
            + ", ".join(duplicates)
        )

    return mapping.set_index("bus")["access_region"]


def load_region_access_shares(path):
    """Load World Bank access-potential shares by access region."""
    if not path:
        return pd.DataFrame.from_dict(
            {
                "South-West": {"grid": 0.52, "isolated": 0.06, "standalone": 0.42},
                "East": {"grid": 0.32, "isolated": 0.10, "standalone": 0.58},
                "North-Center": {
                    "grid": 0.01,
                    "isolated": 0.16,
                    "standalone": 0.83,
                },
            },
            orient="index",
        )
    if not os.path.exists(path):
        raise FileNotFoundError(f"Region access-share file not found: {path}")

    shares = pd.read_csv(path)
    required_cols = {"access_region", "grid", "isolated", "standalone"}
    if not required_cols.issubset(shares.columns):
        raise ValueError(
            "Region access shares must contain columns "
            f"{required_cols}. Found {set(shares.columns)}"
        )

    shares = shares.assign(
        access_region=lambda df: df["access_region"].astype(str).str.strip()
    ).set_index("access_region")
    for col in ["grid", "isolated", "standalone"]:
        shares[col] = pd.to_numeric(shares[col], errors="raise")

    share_sum = shares[["grid", "isolated", "standalone"]].sum(axis=1)
    if not np.allclose(share_sum, 1.0, atol=1e-6):
        invalid = share_sum[~np.isclose(share_sum, 1.0, atol=1e-6)]
        raise ValueError(
            "Region access shares must sum to 1 per region. Invalid sums: "
            + invalid.to_dict().__repr__()
        )
    if ((shares["standalone"] < 0.0) | (shares["standalone"] > 1.0)).any():
        raise ValueError("Standalone access shares must be in [0, 1].")

    return shares


def resolve_access_region(nearest_bus, access_region_by_bus):
    """
    Map a nearest PyPSA bus to an access region.

    DC buses normally use their AC counterpart. DC-only buses can be mapped
    explicitly in the CSV.
    """
    nearest_bus = str(nearest_bus)
    if nearest_bus in access_region_by_bus.index:
        return nearest_bus, access_region_by_bus.loc[nearest_bus]

    ac_equivalent = nearest_bus.replace("_DC_", "_AC_")
    if ac_equivalent in access_region_by_bus.index:
        return ac_equivalent, access_region_by_bus.loc[ac_equivalent]

    raise ValueError(
        "No access region found for nearest bus "
        f"{nearest_bus}. Add it to the access-region mapping CSV."
    )


def identify_standalone_offgrid_regions(
    shapes,
    bus_geom,
    elec_data,
    national_avg,
    cfg,
    access_region_mapping_path,
    region_access_shares_path,
):
    """
    Identify all GADM regions as offgrid candidates and assign standalone population.

    This CD 2035 mode bypasses C1-C5 as selection filters. Distance,
    electrification rate and density are still calculated for reporting.
    """
    access_region_by_bus = load_access_region_mapping(access_region_mapping_path)
    access_shares = load_region_access_shares(region_access_shares_path)

    if "pop" not in shapes.columns:
        raise ValueError("Shapes must contain a 'pop' column for population scaling.")

    shapes = shapes.copy()
    shapes["pop"] = pd.to_numeric(shapes["pop"], errors="raise")
    population_base_total = float(shapes["pop"].sum())
    if population_base_total <= 0.0:
        raise ValueError("Shape population sum must be positive.")

    population_target = cfg.get("population_target")
    if population_target is None:
        population_scale = 1.0
        population_target = population_base_total
    else:
        population_target = float(population_target)
        if population_target <= 0.0:
            raise ValueError("offgrid.population_target must be positive.")
        population_scale = population_target / population_base_total

    offgrid_rows = []
    for _, region in shapes.iterrows():
        gadm_id = region["GADM_ID"]
        centroid = region.geometry.centroid
        bus_distances = bus_geom.geometry.distance(centroid) * 111
        nearest_bus = bus_distances.idxmin()
        mapped_bus, access_region = resolve_access_region(
            nearest_bus, access_region_by_bus
        )

        if access_region not in access_shares.index:
            raise ValueError(
                f"Access region '{access_region}' for {gadm_id} is not present "
                f"in {region_access_shares_path}."
            )

        distance_km = float(bus_distances.loc[nearest_bus])
        area_km2 = region.geometry.area * (111**2)
        population_base = float(region["pop"])
        population_scaled = population_base * population_scale
        standalone_share = float(access_shares.loc[access_region, "standalone"])
        population_offgrid = population_scaled * standalone_share
        pop_density = population_scaled / area_km2 if area_km2 > 0 else 0.0

        elec_rate, elec_src = get_electrification_rate(
            gadm_id, distance_km, pop_density, elec_data, national_avg
        )

        offgrid_rows.append(
            {
                "gadm_id": gadm_id,
                "population": population_offgrid,
                "population_base": population_base,
                "population_scaled": population_scaled,
                "standalone_share": standalone_share,
                "nearest_bus": nearest_bus,
                "mapped_bus": mapped_bus,
                "access_region": access_region,
                "elec_rate": elec_rate,
                "elec_source": elec_src,
                "distance_km": round(distance_km, 1),
                "pop_density": round(pop_density, 1),
                "centroid_x": centroid.x,
                "centroid_y": centroid.y,
                "geometry": region.geometry,
            }
        )

    offgrid_df = pd.DataFrame(offgrid_rows)
    logger.info(
        "Standalone population split: base %.3f Mio, target %.3f Mio, scale %.6f."
        % (population_base_total / 1e6, population_target / 1e6, population_scale)
    )
    logger.info(
        "Standalone offgrid population: %.3f Mio across %d regions."
        % (offgrid_df["population"].sum() / 1e6, len(offgrid_df))
    )
    return offgrid_df


def calibrate_kwh_per_person_to_offgrid_demand(
    cfg, offgrid_df, offgrid_demand_path, access_region_mapping_path
):
    """Set region-specific kWh/person/year from the demand-split output."""
    offgrid_df = offgrid_df.copy()
    if not cfg.get("calibrate_kwh_per_person_to_offgrid_demand", False):
        offgrid_df["kwh_per_person_yr"] = float(cfg["kwh_per_person_yr"])
        return cfg, offgrid_df

    if not offgrid_demand_path or not os.path.exists(offgrid_demand_path):
        raise FileNotFoundError(
            "Offgrid demand profile for kWh/person calibration not found: "
            f"{offgrid_demand_path}"
        )

    if "access_region" not in offgrid_df.columns:
        raise ValueError("offgrid_df must contain access_region for regional calibration.")

    population_by_region = offgrid_df.groupby("access_region")["population"].sum()
    if (population_by_region <= 0.0).any():
        invalid = population_by_region[population_by_region <= 0.0].index
        raise ValueError(
            "Standalone offgrid population must be positive per access region: "
            + ", ".join(map(str, invalid))
        )

    offgrid_demand = pd.read_csv(offgrid_demand_path, index_col=0).apply(
        pd.to_numeric, errors="raise"
    )
    timestep_hours = profile_timestep_hours(offgrid_demand.index)
    target_by_bus_mwh = offgrid_demand.sum(axis=0) * timestep_hours
    target_by_bus_mwh.index = target_by_bus_mwh.index.astype(str)

    access_region_by_bus = load_access_region_mapping(access_region_mapping_path)
    missing_buses = target_by_bus_mwh.index.difference(access_region_by_bus.index)
    positive_missing = target_by_bus_mwh.reindex(missing_buses).dropna()
    positive_missing = positive_missing[positive_missing > 1e-6]
    if not positive_missing.empty:
        raise ValueError(
            "Offgrid demand profile contains buses without access-region mapping: "
            + ", ".join(map(str, positive_missing.index[:10]))
        )

    target_regions = access_region_by_bus.reindex(target_by_bus_mwh.index)
    target_by_region_mwh = target_by_bus_mwh.groupby(target_regions).sum()
    target_mwh = float(target_by_region_mwh.sum())
    if target_mwh <= 0.0:
        raise ValueError("Offgrid demand target must be positive.")

    missing_population = target_by_region_mwh[
        target_by_region_mwh > 1e-6
    ].index.difference(population_by_region.index)
    if len(missing_population):
        raise ValueError(
            "Offgrid demand target has access regions without standalone population: "
            + ", ".join(map(str, missing_population))
        )

    target_by_region_mwh = target_by_region_mwh.reindex(population_by_region.index)
    missing_or_zero_target = target_by_region_mwh[
        target_by_region_mwh.isna() | (target_by_region_mwh <= 0.0)
    ]
    if not missing_or_zero_target.empty:
        raise ValueError(
            "Standalone access regions need positive offgrid demand targets: "
            + ", ".join(map(str, missing_or_zero_target.index))
        )

    kwh_by_region = target_by_region_mwh * 1000.0 / population_by_region
    offgrid_df["kwh_per_person_yr"] = offgrid_df["access_region"].map(kwh_by_region)
    if offgrid_df["kwh_per_person_yr"].isna().any():
        raise ValueError("Failed to assign regional kWh/person/year to all offgrid rows.")

    cfg["kwh_per_person_yr"] = target_mwh * 1000.0 / float(population_by_region.sum())
    cfg["kwh_per_person_yr_by_access_region"] = kwh_by_region.to_dict()
    cfg["offgrid_demand_target_mwh"] = target_mwh
    cfg["offgrid_demand_target_mwh_by_access_region"] = target_by_region_mwh.to_dict()

    logger.info(
        "Calibrated regional kWh/person/year from %.6f TWh target "
        "(profile step %.2fh): %s"
        % (
            target_mwh / 1e6,
            timestep_hours,
            ", ".join(
                f"{region}={value:.3f}"
                for region, value in kwh_by_region.sort_index().items()
            ),
        )
    )
    return cfg, offgrid_df


# ══════════════════════════════════════════════════════
# HILFSFUNKTIONEN – Snapshot-Gewichtung
# ══════════════════════════════════════════════════════


def profile_timestep_hours(index):
    """Infer the time step of a demand profile in hours."""
    timestamps = pd.to_datetime(index, errors="coerce")
    if pd.isna(timestamps).any():
        return 1.0

    timestamps = pd.DatetimeIndex(timestamps)
    diffs = timestamps.to_series().diff().dropna()
    if diffs.empty:
        return 1.0

    timestep_hours = diffs.median().total_seconds() / 3600
    if not np.isfinite(timestep_hours) or timestep_hours <= 0.0:
        return 1.0
    return float(timestep_hours)


def get_snapshot_weightings(n, snapshots):
    """Return generator snapshot weights aligned to the network snapshots."""
    if hasattr(n, "snapshot_weightings") and "generators" in n.snapshot_weightings:
        weights = n.snapshot_weightings["generators"].reindex(snapshots)
    else:
        weights = pd.Series(1.0, index=snapshots)

    weights = pd.to_numeric(weights, errors="coerce").fillna(1.0)
    if (weights <= 0.0).any():
        raise ValueError("Snapshot weightings must be positive.")
    return weights


def normalize_weighted_profile(values, weights):
    """Scale a profile to weighted mean 1.0."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    weighted_mean = np.average(values, weights=weights)
    if not np.isfinite(weighted_mean) or weighted_mean <= 0.0:
        raise ValueError("Profile weighted mean must be positive.")
    return values / weighted_mean


def weighted_timeseries_sum(series, weights):
    """Sum a PyPSA time series in MWh using snapshot weights."""
    aligned = pd.to_numeric(series.reindex(weights.index), errors="raise")
    if aligned.isna().any():
        raise ValueError("Time series cannot be aligned to snapshot weightings.")
    return float(aligned.mul(weights).sum())


# ══════════════════════════════════════════════════════
# HILFSFUNKTIONEN – Solar Profile
# ══════════════════════════════════════════════════════


def snapshot_hours(snapshots):
    """Return hour-of-day values for PyPSA snapshots."""
    index = pd.Index(snapshots)
    timestamps = pd.to_datetime(index, errors="coerce")
    if not pd.isna(timestamps).any():
        return pd.DatetimeIndex(timestamps).hour.to_numpy()
    return np.arange(len(index)) % 24


def get_solar_profile_from_era5(centroid_x, centroid_y, cutout, snapshots):
    """
    Echtes Solarprofil aus ERA5-Wetterdaten.
    Quelle: Hersbach et al. (2020), doi:10.1002/qj.3803
    """
    cutout_point = cutout.sel(
        x=slice(centroid_x - 0.3, centroid_x + 0.3),
        y=slice(centroid_y - 0.3, centroid_y + 0.3),
    )
    influx = cutout_point.data["influx_direct"] + cutout_point.data["influx_diffuse"]
    influx_series = influx.mean(dim=["x", "y"]).to_series()
    influx_series.index = pd.to_datetime(influx_series.index)

    target_index = pd.to_datetime(pd.Index(snapshots), errors="coerce")
    if pd.isna(target_index).any():
        influx_ts = influx_series.to_numpy()[: len(snapshots)]
    else:
        influx_ts = influx_series.reindex(pd.DatetimeIndex(target_index)).to_numpy()
        if np.isnan(influx_ts).any():
            influx_ts = influx_series.to_numpy()[: len(snapshots)]

    if len(influx_ts) != len(snapshots):
        raise ValueError("ERA5 solar profile length does not match network snapshots.")

    max_val = influx_ts.max()
    if max_val > 0:
        return influx_ts / max_val
    return solar_profile_fallback(snapshots)


def load_profile(snapshots):
    """Tagesprofil fuer laendliche Haushalte."""
    hod = snapshot_hours(snapshots)
    p = np.ones(len(hod)) * 0.6
    p[hod >= 6] = 0.8
    p[hod >= 9] = 0.6
    p[hod >= 17] = 1.0
    p[hod >= 21] = 0.7
    p[hod >= 23] = 0.4
    return p / p.mean()


def solar_profile_fallback(snapshots):
    """Fallback Solarprofil falls ERA5 nicht verfuegbar."""
    hod = snapshot_hours(snapshots)
    s = np.zeros(len(hod))
    s[hod >= 6] = 0.3
    s[hod >= 9] = 0.7
    s[hod >= 11] = 0.9
    s[hod >= 13] = 0.8
    s[hod >= 16] = 0.4
    s[hod >= 18] = 0.0
    return s


# ══════════════════════════════════════════════════════
# HILFSFUNKTIONEN – Geo-Logik
# ══════════════════════════════════════════════════════


def identify_offgrid_counties(
    shapes, bus_geom, bus_load_mw, bus_max_line_loading, elec_data, national_avg, cfg
):
    """
    Layer 1 – Geo-Logik: Identifiziert Offgrid-Counties.

    Prueft Kriterien C1-C5 fuer jede Region in shapes.
    Alle Kriterien konfigurierbar via config.yaml unter 'offgrid:'.

    C1: Netzabstand > max_distance_km
    C2: Elektrifizierungsrate < max_elec_rate
    C3: Bevoelkerungsdichte > min_pop_density
    C4: Baseline-Last < c4_max_load_mw  (optional)
    C5: Leitungsauslastung > c5_line_loading  (Szenario C, optional)

    Returns:
        DataFrame mit Offgrid-Counties
    """
    offgrid_rows = []

    for _, region in shapes.iterrows():
        gadm_id = region["GADM_ID"]
        centroid = region.geometry.centroid
        dist_km = (bus_geom.geometry.distance(centroid) * 111).min()
        area_km2 = region.geometry.area * (111**2)
        pop_density = region["pop"] / area_km2 if area_km2 > 0 else 0

        elec_rate, elec_src = get_electrification_rate(
            gadm_id, dist_km, pop_density, elec_data, national_avg
        )

        # C1: Netzabstand
        c1 = (dist_km > cfg["max_distance_km"]) if cfg["use_c1_distance"] else True
        # C2: Elektrifizierungsrate
        c2 = (elec_rate < cfg["max_elec_rate"]) if cfg["use_c2_elec_rate"] else True
        # C3: Bevoelkerungsdichte
        c3 = (
            (pop_density > cfg["min_pop_density"])
            if cfg["use_c3_pop_density"]
            else True
        )
        # C4: Baseline-Last zu klein → Region zu klein fuer Hauptnetz
        nearest_bus = bus_geom.geometry.distance(centroid).idxmin()
        nearest_load = bus_load_mw.get(nearest_bus, 0.0)
        c4 = (nearest_load < cfg["c4_max_load_mw"]) if cfg["use_c4_low_load"] else True
        # C5: Leitungsauslastung (Szenario C – Offgrid entlastet Netz)
        nearest_loading = bus_max_line_loading.get(nearest_bus, 0.0)
        c5 = (
            (nearest_loading > cfg["c5_line_loading"])
            if cfg["use_c5_congestion"]
            else True
        )

        if c1 and c2 and c3 and c4 and c5:
            offgrid_rows.append(
                {
                    "gadm_id": gadm_id,
                    "population": int(region["pop"]),
                    "elec_rate": elec_rate,
                    "elec_source": elec_src,
                    "distance_km": round(dist_km, 1),
                    "pop_density": round(pop_density, 1),
                    "centroid_x": centroid.x,
                    "centroid_y": centroid.y,
                    "geometry": region.geometry,
                }
            )

    return pd.DataFrame(offgrid_rows)


# ══════════════════════════════════════════════════════
# HILFSFUNKTIONEN – Mini-Grid hinzufügen
# ══════════════════════════════════════════════════════


def add_offgrid_bus(
    n,
    region_id,
    population,
    kwh_per_person_yr,
    centroid_x,
    centroid_y,
    cfg,
    cutout,
    annuity_factor,
    snapshots,
    snapshot_weightings,
    offgrid_costs,
):
    """
    Fuegt einen isolierten Offgrid-Bus + Komponenten zum Netzwerk hinzu.

    PyPSA Einheiten (Quelle: docs.pypsa.org):
        - p_set / p_nom:  MW
        - capital_cost:   EUR/MW/a
        - marginal_cost:  EUR/MWh

    Kein Link zum Hauptnetz → vollstaendig isoliert.
    Offgrid capital_cost values are annualized PyPSA costs in EUR/MW/a.
    """
    bus_name = f"offgrid_{region_id}"

    if bus_name in n.buses.index:
        logger.warning(f"Bus {bus_name} bereits vorhanden – ueberspringe")
        return None

    annual_hours = float(snapshot_weightings.sum())
    if annual_hours <= 0.0:
        raise ValueError("Sum of snapshot weightings must be positive.")

    # ✅ Isolierten Bus hinzufuegen (kein Link zum Hauptnetz)
    n.add(
        "Bus", bus_name, carrier="offgrid-AC", x=centroid_x, y=centroid_y, v_nom=0.4
    )  # 400V Niederspannung – typisch fuer Minigrids

    # Last in MW: annual MWh / weighted annual hours.
    avg_load_mw = population * kwh_per_person_yr / 1000 / annual_hours
    lp = normalize_weighted_profile(
        load_profile(snapshots), snapshot_weightings
    ) * avg_load_mw
    lts = pd.Series(lp, index=snapshots)
    n.add("Load", f"load_{region_id}", bus=bus_name, p_set=lts)

    # Solar Profil – ERA5 oder Fallback
    if cutout is not None:
        try:
            sp = get_solar_profile_from_era5(
                centroid_x, centroid_y, cutout, snapshots
            )
            solar_src = "ERA5"
        except Exception as e:
            logger.warning(f"ERA5 Fehler fuer {region_id}: {e} → Fallback")
            sp = solar_profile_fallback(snapshots)
            solar_src = "Fallback"
    else:
        sp = solar_profile_fallback(snapshots)
        solar_src = "Fallback"

    sts = pd.Series(sp, index=snapshots)

    # Solar: PyPSA-derived annualized capital cost in EUR/MW/a.
    n.add(
        "Generator",
        f"solar_{region_id}",
        bus=bus_name,
        carrier="solar",
        p_nom_extendable=True,
        p_nom_max=float("inf"),
        p_max_pu=sts,
        capital_cost=offgrid_costs["solar_capital_cost_eur_per_mw_a"],
        marginal_cost=offgrid_costs["solar_marginal_cost_eur_per_mwh"],
    )

    # Battery: PyPSA-derived annualized 6h battery cost in EUR/MW/a.
    n.add(
        "StorageUnit",
        f"battery_{region_id}",
        bus=bus_name,
        carrier="battery",
        p_nom_extendable=True,
        max_hours=offgrid_costs["battery_max_hours"],
        capital_cost=offgrid_costs["battery_capital_cost_eur_per_mw_a"],
        marginal_cost=offgrid_costs["battery_marginal_cost_eur_per_mwh"],
        efficiency_store=offgrid_costs["battery_efficiency_store"],
        efficiency_dispatch=offgrid_costs["battery_efficiency_dispatch"],
        cyclic_state_of_charge=True,
    )

    # Diesel backup uses PyPSA oil costs as diesel/oil proxy.
    n.add(
        "Generator",
        f"diesel_{region_id}",
        bus=bus_name,
        carrier="diesel",
        p_nom_extendable=True,
        capital_cost=offgrid_costs["diesel_capital_cost_eur_per_mw_a"],
        marginal_cost=offgrid_costs["diesel_marginal_cost_eur_per_mwh"],
        efficiency=offgrid_costs["diesel_efficiency"],
    )

    # Load Shedding – Sicherheit
    n.add(
        "Generator",
        f"shedding_{region_id}",
        bus=bus_name,
        carrier="load_shedding",
        p_nom=1e6,
        p_nom_extendable=False,
        marginal_cost=offgrid_costs["load_shedding_marginal_cost_eur_per_mwh"],
    )

    logger.info(f"  {region_id}: Bus + Solar + Batterie + Diesel [{solar_src}]")
    return solar_src


# ══════════════════════════════════════════════════════
# HILFSFUNKTIONEN – Offgrid-only Optimierung
# ══════════════════════════════════════════════════════


def create_offgrid_optimization_network(n_base):
    """
    Erzeugt ein reines Offgrid-Netz.

    Das geloeste On-grid-Netz wird nur als Vorlage fuer Snapshots,
    Snapshot-Gewichtungen und Carrier-Metadaten verwendet.
    """
    n_off = pypsa.Network()
    n_off.set_snapshots(n_base.snapshots)

    if hasattr(n_base, "snapshot_weightings"):
        n_off.snapshot_weightings = n_base.snapshot_weightings.copy()

    for carrier in ["offgrid-AC", "solar", "battery", "diesel", "load_shedding"]:
        if carrier in n_base.carriers.index:
            n_off.import_components_from_dataframe(
                n_base.carriers.loc[[carrier]].copy(), "Carrier"
            )
        elif carrier not in n_off.carriers.index:
            n_off.add("Carrier", carrier)

    return n_off


def merge_offgrid_results_into_base(n_base, n_off):
    """
    Fuegt optimierte Offgrid-Komponenten in eine Kopie des Basisnetzes ein.

    Bestehende On-grid-Komponenten und ihre Zeitreihen bleiben unveraendert.
    """
    n_out = n_base.copy()

    for carrier in n_off.carriers.index.difference(n_out.carriers.index):
        n_out.import_components_from_dataframe(
            n_off.carriers.loc[[carrier]].copy(), "Carrier"
        )

    for cls_name in ["Bus", "Load", "Generator", "StorageUnit", "Store", "Link"]:
        off_components = n_off.df(cls_name)
        if off_components.empty:
            continue

        new_components = off_components.index.difference(n_out.df(cls_name).index)
        if new_components.empty:
            continue

        n_out.import_components_from_dataframe(
            off_components.loc[new_components].copy(), cls_name
        )

        for attr, series_df in n_off.pnl(cls_name).items():
            if not isinstance(series_df, pd.DataFrame) or series_df.empty:
                continue
            columns = series_df.columns.intersection(new_components)
            if columns.empty:
                continue
            n_out.import_series_from_dataframe(
                series_df.loc[:, columns].copy(), cls_name, attr
            )

    return n_out


# ══════════════════════════════════════════════════════
# HILFSFUNKTIONEN – Ergebnisse auswerten
# ══════════════════════════════════════════════════════


def extract_results(
    n, offgrid_df, cfg, annuity_factor, snapshot_weightings, offgrid_costs
):
    """
    Liest Optimierungsergebnisse fuer alle Offgrid-Busse aus.

    LCOE:    total_cost_yr / total_load_mwh / 1000  [EUR/kWh]
    Autarkie: solar_gen / (solar_gen + diesel_gen)   [%]
    CO2:     diesel_gen_mwh * PyPSA oil emissions    [t/yr]
    """
    rows = []
    for _, row in offgrid_df.iterrows():
        rid = row["gadm_id"]
        try:
            solar_cap_mw = n.generators.loc[f"solar_{rid}", "p_nom_opt"]
            battery_cap_mw = n.storage_units.loc[f"battery_{rid}", "p_nom_opt"]
            diesel_cap_mw = n.generators.loc[f"diesel_{rid}", "p_nom_opt"]
            solar_gen_mwh = weighted_timeseries_sum(
                n.generators_t.p[f"solar_{rid}"], snapshot_weightings
            )
            diesel_gen_mwh = weighted_timeseries_sum(
                n.generators_t.p[f"diesel_{rid}"], snapshot_weightings
            )
            shedding_gen_mwh = (
                weighted_timeseries_sum(
                    n.generators_t.p[f"shedding_{rid}"], snapshot_weightings
                )
                if f"shedding_{rid}" in n.generators_t.p.columns
                else 0.0
            )

            # Kapazitaeten in kW fuer CSV
            solar_cap_kw = solar_cap_mw * 1000
            battery_cap_kw = battery_cap_mw * 1000
            diesel_cap_kw = diesel_cap_mw * 1000

            annualized_capital_cost_yr = (
                solar_cap_mw * offgrid_costs["solar_capital_cost_eur_per_mw_a"]
                + battery_cap_mw * offgrid_costs["battery_capital_cost_eur_per_mw_a"]
                + diesel_cap_mw * offgrid_costs["diesel_capital_cost_eur_per_mw_a"]
            )
            variable_cost_yr = (
                solar_gen_mwh * offgrid_costs["solar_marginal_cost_eur_per_mwh"]
                + diesel_gen_mwh * offgrid_costs["diesel_marginal_cost_eur_per_mwh"]
                + shedding_gen_mwh
                * offgrid_costs["load_shedding_marginal_cost_eur_per_mwh"]
            )
            total_cost_yr = annualized_capital_cost_yr + variable_cost_yr

            kwh_per_person_yr = float(
                row.get("kwh_per_person_yr", cfg["kwh_per_person_yr"])
            )

            # Gesamtlast [MWh/yr]
            total_load_mwh = row["population"] * kwh_per_person_yr / 1000

            # ✅ LCOE [EUR/kWh]
            lcoe = total_cost_yr / total_load_mwh / 1000

            # ✅ Autarkie [%] – Solar-Anteil an Gesamterzeugung (max 100%)
            total_gen_mwh = solar_gen_mwh + diesel_gen_mwh
            autarky = (
                (solar_gen_mwh / total_gen_mwh * 100) if total_gen_mwh > 0 else 0.0
            )

            # ✅ CO2 [t/yr]
            co2_t = diesel_gen_mwh * offgrid_costs["diesel_co2_t_per_mwh_el"]

            # Netzanschluss Kosten (zum Vergleich)
            grid_capex = row["distance_km"] * 15000 + 35000
            grid_total_yr = grid_capex * annuity_factor + grid_capex * 0.03
            offgrid_cheaper = total_cost_yr < grid_total_yr

            rows.append(
                {
                    "region": rid,
                    "population": row["population"],
                    "population_base": row.get("population_base", row["population"]),
                    "population_scaled": row.get("population_scaled", row["population"]),
                    "standalone_share": row.get("standalone_share", np.nan),
                    "kwh_per_person_yr": round(kwh_per_person_yr, 3),
                    "nearest_bus": row.get("nearest_bus", ""),
                    "access_region": row.get("access_region", ""),
                    "distance_km": row["distance_km"],
                    "elec_rate": row["elec_rate"],
                    "elec_source": row["elec_source"],
                    "solar_kw": round(solar_cap_kw, 1),
                    "battery_kw": round(battery_cap_kw, 1),
                    "battery_kwh": round(
                        battery_cap_kw * offgrid_costs["battery_max_hours"], 1
                    ),
                    "diesel_kw": round(diesel_cap_kw, 1),
                    "capex_total_keur": round(annualized_capital_cost_yr / 1000, 1),
                    "annualized_capital_cost_keur_yr": round(
                        annualized_capital_cost_yr / 1000, 1
                    ),
                    "variable_cost_keur_yr": round(variable_cost_yr / 1000, 1),
                    "total_cost_keur_yr": round(total_cost_yr / 1000, 1),
                    "grid_capex_keur": round(grid_capex / 1000, 1),
                    "grid_total_keur_yr": round(grid_total_yr / 1000, 1),
                    "offgrid_cheaper": offgrid_cheaper,
                    "lcoe_eur_kwh": round(lcoe, 3),
                    "autarky_pct": round(autarky, 1),
                    "co2_t_yr": round(co2_t, 2),
                    "centroid_x": row["centroid_x"],
                    "centroid_y": row["centroid_y"],
                }
            )
        except Exception as e:
            logger.warning(f"Auswertung fehlgeschlagen fuer {rid}: {e}")

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════
# HAUPTPROGRAMM
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    # ── Snakemake oder direkt aufrufen ────────────────────────────────
    if "snakemake" not in dir():
        import glob
        import subprocess
        import sys
        from pathlib import Path

        import yaml
        from _helpers import (
            resolve_config_by_planning_horizon,
            resolve_h2export_for_planning_horizon,
        )

        config_path = Path("configs/scenarios_H2G/config.H2G_A_CD_myopic.yaml")
        with config_path.open() as fh:
            base_config = yaml.safe_load(fh) or {}

        run_name = base_config.get("run", {}).get("sector_name") or base_config.get(
            "run", {}
        ).get("name", "H2G_A_CD_myopic")
        planning_horizons = [
            int(year) for year in base_config["scenario"]["planning_horizons"]
        ]
        population_targets = {
            2025: 112830000,
            2035: 151560000,
            2050: 218250000,
        }

        def labels_for_horizon(year):
            labels_by_horizon = (
                base_config.get("export", {})
                .get("h2export_by_planning_horizon_labels", {})
                or {}
            )
            labels = labels_by_horizon.get(year, labels_by_horizon.get(str(year)))
            if labels is None:
                labels = base_config.get("export", {}).get("h2export", [0])
                if not isinstance(labels, list):
                    labels = [labels]
                selected = []
                seen = set()
                for label in labels:
                    quantity = resolve_h2export_for_planning_horizon(
                        base_config, label, year
                    )
                    quantity_key = round(float(quantity), 9)
                    if quantity_key in seen:
                        continue
                    seen.add(quantity_key)
                    selected.append(label)
                labels = selected
            return [str(label) for label in labels]

        all_scenarios = [
            (year, label)
            for year in planning_horizons
            for label in labels_for_horizon(year)
        ]

        def arg_value(name, default=None):
            if name not in sys.argv:
                return default
            arg_pos = sys.argv.index(name)
            try:
                return sys.argv[arg_pos + 1]
            except IndexError as exc:
                raise SystemExit(f"{name} needs one value.") from exc

        direct_year = arg_value("--year", os.environ.get("H2G_OFFGRID_YEAR"))
        direct_export = os.environ.get("H2G_OFFGRID_H2EXPORT")
        direct_export = arg_value("--h2export", direct_export)
        list_only = "--list" in sys.argv or "--dry-run" in sys.argv

        if direct_year in {None, "all"} and direct_export in {None, "all"}:
            requested_scenarios = all_scenarios
        elif direct_year in {None, "all"}:
            requested_scenarios = [
                scenario for scenario in all_scenarios if scenario[1] == direct_export
            ]
        elif direct_export in {None, "all"}:
            year_int = int(direct_year)
            requested_scenarios = [
                scenario for scenario in all_scenarios if scenario[0] == year_int
            ]
        else:
            requested_scenarios = [(int(direct_year), str(direct_export))]

        valid_scenarios = set(all_scenarios)
        invalid_scenarios = [
            scenario for scenario in requested_scenarios if scenario not in valid_scenarios
        ]
        if invalid_scenarios:
            valid_text = ", ".join(
                f"{year}:{label}" for year, label in all_scenarios
            )
            invalid_text = ", ".join(
                f"{year}:{label}" for year, label in invalid_scenarios
            )
            raise SystemExit(
                f"Unsupported myopic offgrid scenario(s): {invalid_text}. "
                f"Valid scenarios: {valid_text}."
            )

        sopts_values = base_config.get("scenario", {}).get("sopts", ["3H"])
        if not isinstance(sopts_values, list):
            sopts_values = [sopts_values]
        sopts_label = str(sopts_values[0])

        def network_stem(year, h2export_label):
            return (
                f"elec_s_all_ec_lcopt_CCL_{sopts_label}_"
                f"{year}_0.1046_NZ_{h2export_label}export"
            )

        def network_path(year, h2export_label):
            return (
                Path("results")
                / run_name
                / "postnetworks"
                / f"{network_stem(year, h2export_label)}.nc"
            )

        def offgrid_network_path(year, h2export_label):
            return (
                Path("results")
                / run_name
                / "offgrid"
                / "postnetworks"
                / f"{network_stem(year, h2export_label)}_offgrid.nc"
            )

        def offgrid_csv_path(year, h2export_label):
            return (
                Path("results")
                / run_name
                / "offgrid"
                / f"offgrid_results_{network_stem(year, h2export_label)}.csv"
            )

        if list_only:
            for year, export_label in requested_scenarios:
                print(
                    f"{year}:{export_label} -> "
                    f"{offgrid_network_path(year, export_label)}"
                )
            raise SystemExit(0)

        missing_networks = [
            network_path(year, label)
            for year, label in requested_scenarios
            if not network_path(year, label).exists()
        ]
        if missing_networks:
            raise FileNotFoundError(
                "Missing solved myopic postnetworks for direct offgrid run:\n"
                + "\n".join(str(path) for path in missing_networks)
            )

        if direct_year in {None, "all"} or direct_export in {None, "all"}:
            logger.info(
                "Direkter Aufruf – starte Myopic-Offgrid fuer %d Varianten: %s",
                len(requested_scenarios),
                ", ".join(f"{year}:{label}" for year, label in requested_scenarios),
            )
            for year, export_label in requested_scenarios:
                logger.info("Starte Myopic-Offgrid %s | %sexport", year, export_label)
                env = {
                    **os.environ,
                    "H2G_OFFGRID_YEAR": str(year),
                    "H2G_OFFGRID_H2EXPORT": export_label,
                }
                subprocess.run(
                    [
                        sys.executable,
                        __file__,
                        "--year",
                        str(year),
                        "--h2export",
                        export_label,
                    ],
                    check=True,
                    env=env,
                )
            raise SystemExit(0)

        year_int = int(direct_year)
        h2export_label = str(direct_export)
        direct_config = resolve_config_by_planning_horizon(base_config, year_int)
        direct_config.setdefault("scenario", {})["planning_horizons"] = [year_int]
        direct_config.setdefault("export", {})["h2export"] = [
            resolve_h2export_for_planning_horizon(
                base_config, h2export_label, year_int
            )
        ]
        direct_network_stem = network_stem(year_int, h2export_label)
        direct_network_path = network_path(year_int, h2export_label)
        horizon_resource_dir = Path("resources") / run_name / f"horizon_{year_int}"
        if not direct_network_path.exists():
            raise FileNotFoundError(
                "Missing solved myopic postnetwork for direct offgrid run: "
                f"{direct_network_path}"
            )

        class MockSnakemake:
            class input:
                network = str(direct_network_path)
                shapes = f"resources/{run_name}/shapes/gadm_shapes.geojson"
                cutout = "cutouts/cutout-2013-era5.nc"
                # Optional: falls nicht vorhanden, nutzt das Skript die Heuristik.
                elec_data = glob.glob("data/elec_rates/CD_electricity_access.csv")
                access_region_mapping = direct_config["load_options"][
                    "demand_allocation"
                ]["bus_region_mapping"]
                region_access_shares = ""
                offgrid_demand = str(horizon_resource_dir / "offgrid_demand_profiles.csv")
                tech_costs = f"resources/{run_name}/costs_{year_int}.csv"

            class output:
                network = str(offgrid_network_path(year_int, h2export_label))
                csv = str(offgrid_csv_path(year_int, h2export_label))

            class log:
                python = (
                    f"logs/{run_name}/build_offgrid_"
                    f"{year_int}_{h2export_label}export.log"
                )

            config = {
                **direct_config,
                "offgrid": {
                    "enable": True,
                    "use_c1_distance": True,
                    "use_c2_elec_rate": True,
                    "use_c3_pop_density": True,
                    "use_c4_low_load": False,
                    "use_c5_congestion": False,
                    "max_distance_km": 50,
                    "max_elec_rate": 0.50,
                    "min_pop_density": 10,
                    "c4_max_load_mw": 20.0,
                    "c5_line_loading": 0.50,
                    "kwh_per_person_yr": 60,
                    "population_target": population_targets.get(year_int, 218250000),
                    "use_standalone_population_split": True,
                    "calibrate_kwh_per_person_to_offgrid_demand": True,
                    "restrict_to_ac_buses": True,
                    "restrict_to_ac_loads": True,
                    "solver": "gurobi",
                    "solver_options": {
                        "threads": 16,
                        "method": 2,
                        "crossover": 0,
                    },
                    "discount_rate": 0.1046,
                    "asset_lifetime": 20,
                }
            }

        snakemake = MockSnakemake()
        logger.info("Direkter Aufruf – MockSnakemake aktiv")

    # ── Config lesen ──────────────────────────────────────────────────
    cfg = dict(snakemake.config["offgrid"])

    # Offgrid Feature an/aus
    if not cfg.get("enable", True):
        logger.info("Offgrid deaktiviert (enable: false) → unveraendert kopieren")
        n = pypsa.Network(snakemake.input.network)
        os.makedirs(os.path.dirname(snakemake.output.network), exist_ok=True)
        os.makedirs(os.path.dirname(snakemake.output.csv), exist_ok=True)
        n.export_to_netcdf(snakemake.output.network)
        pd.DataFrame().to_csv(snakemake.output.csv, index=False)
        exit(0)

    logger.info("=" * 60)
    logger.info("BUILD OFFGRID – Offgrid-Integration in PyPSA-Earth")
    logger.info("=" * 60)

    # ── Annuitätsfaktor ───────────────────────────────────────────────
    # Quelle: ESMAP Mini Grid Design Manual (2019)
    _r = cfg["discount_rate"]
    _n = cfg["asset_lifetime"]
    annuity_factor = _r * (1 + _r) ** _n / ((1 + _r) ** _n - 1)
    logger.info(
        f"Annuitätsfaktor: {annuity_factor:.4f} " f"(r={_r*100:.0f}%, n={_n}yr)"
    )

    # ── Schritt 1: Netzwerk laden ─────────────────────────────────────
    logger.info(f"Lade Netzwerk: {snakemake.input.network}")
    n = pypsa.Network(snakemake.input.network)
    snapshots = n.snapshots
    n_hours = len(snapshots)
    logger.info(f"  {len(n.buses)} Busse | {n_hours} Zeitschritte")
    snapshot_weightings = get_snapshot_weightings(n, snapshots)
    logger.info(
        "  Snapshot-Gewichtung: %.1f gewichtete Stunden"
        % snapshot_weightings.sum()
    )
    offgrid_costs = build_offgrid_costs(n, snakemake, snakemake.config)

    if cfg.get("restrict_to_ac_buses", False):
        bus_carrier = n.buses.get("carrier", pd.Series("", index=n.buses.index))
        buses = n.buses.loc[bus_carrier.fillna("").eq("AC"), ["x", "y"]].copy()
        if buses.empty:
            raise ValueError("AC bus filter is active, but no AC buses were found.")
        logger.info(
            "  AC-Bus-Filter aktiv: %d von %d Bussen fuer Offgrid-Zuordnung"
            % (len(buses), len(n.buses))
        )
    else:
        buses = n.buses[["x", "y"]].copy()

    bus_geom = gpd.GeoDataFrame(
        buses, geometry=gpd.points_from_xy(buses.x, buses.y), crs="EPSG:4326"
    )

    # C4: Last pro Bus
    bus_load_mw = {}
    if len(n.loads_t.p_set.columns) > 0:
        load_names = n.loads.index
        if cfg.get("restrict_to_ac_loads", False):
            load_carrier = n.loads.get("carrier", pd.Series("", index=n.loads.index))
            load_names = n.loads.index[load_carrier.fillna("").eq("AC")]
            logger.info(
                "  AC-Load-Filter aktiv: %d von %d Loads"
                % (len(load_names), len(n.loads))
            )

        load_cols = n.loads_t.p_set.columns.intersection(load_names)
        avg_load = (
            n.loads_t.p_set.loc[:, load_cols]
            .multiply(snapshot_weightings, axis=0)
            .sum()
            / snapshot_weightings.sum()
        )
        for load_name in avg_load.index:
            bus_name = n.loads.loc[load_name, "bus"]
            bus_load_mw[bus_name] = bus_load_mw.get(bus_name, 0) + avg_load[load_name]

    # C5: Leitungsauslastung
    bus_max_line_loading = {}
    if len(n.lines_t.p0.columns) > 0 and len(n.lines) > 0:
        s_nom = n.lines.s_nom_opt if "s_nom_opt" in n.lines.columns else n.lines.s_nom
        line_loading = (n.lines_t.p0.abs() / s_nom).max()
        for line_name, loading in line_loading.items():
            for bus in [n.lines.loc[line_name, "bus0"], n.lines.loc[line_name, "bus1"]]:
                bus_max_line_loading[bus] = max(
                    bus_max_line_loading.get(bus, 0.0), loading
                )

    # ── Schritt 2: Shapes laden ───────────────────────────────────────
    logger.info(f"Lade Shapes: {snakemake.input.shapes}")
    shapes = gpd.read_file(snakemake.input.shapes)
    logger.info(f"  {len(shapes)} Regionen")

    # ── Schritt 3: Elektrifizierungsdaten laden ───────────────────────
    # Generisch: alle CSV-Dateien in snakemake.input.elec_data laden
    logger.info("Lade Elektrifizierungsdaten...")
    elec_data, national_avg = load_all_electrification_data(
        get_named_value(snakemake.input, "elec_data", [])
    )

    # ── Schritt 4: ERA5 Cutout laden ─────────────────────────────────
    cutout = None
    if os.path.exists(snakemake.input.cutout):
        try:
            cutout = atlite.Cutout(snakemake.input.cutout)
            logger.info(f"ERA5 geladen ✅")
        except Exception as e:
            logger.warning(f"ERA5 Fehler: {e} → Fallback")
    else:
        logger.warning("ERA5 nicht gefunden → Fallback")

    # ── Schritt 5: Geo-Logik ──────────────────────────────────────────
    if cfg.get("use_standalone_population_split", False):
        logger.info("Geo-Logik: Standalone-Bevoelkerung je GADM-Region ableiten...")
        logger.info("  C1-C5 Filter: INAKTIV in Standalone-Population-Modus")
        offgrid_df = identify_standalone_offgrid_regions(
            shapes=shapes,
            bus_geom=bus_geom,
            elec_data=elec_data,
            national_avg=national_avg,
            cfg=cfg,
            access_region_mapping_path=get_named_value(
                snakemake.input, "access_region_mapping", ""
            ),
            region_access_shares_path=get_named_value(
                snakemake.input, "region_access_shares", ""
            ),
        )
        cfg, offgrid_df = calibrate_kwh_per_person_to_offgrid_demand(
            cfg=cfg,
            offgrid_df=offgrid_df,
            offgrid_demand_path=get_named_value(snakemake.input, "offgrid_demand", ""),
            access_region_mapping_path=get_named_value(
                snakemake.input, "access_region_mapping", ""
            ),
        )
    else:
        logger.info("Geo-Logik: Offgrid-Counties identifizieren...")
        logger.info(
            f"  C1 (Abstand > {cfg['max_distance_km']} km):    "
            f"{'aktiv' if cfg['use_c1_distance'] else 'INAKTIV'}"
        )
        logger.info(
            f"  C2 (Elec < {cfg['max_elec_rate']*100:.0f}%):         "
            f"{'aktiv' if cfg['use_c2_elec_rate'] else 'INAKTIV'}"
        )
        logger.info(
            f"  C3 (Dichte > {cfg['min_pop_density']} P/km2):  "
            f"{'aktiv' if cfg['use_c3_pop_density'] else 'INAKTIV'}"
        )
        logger.info(
            f"  C4 (Last < {cfg['c4_max_load_mw']} MW):        "
            f"{'aktiv' if cfg['use_c4_low_load'] else 'INAKTIV'}"
        )
        logger.info(
            f"  C5 (Auslastung > {cfg['c5_line_loading']*100:.0f}%): "
            f"{'aktiv (Szenario C)' if cfg['use_c5_congestion'] else 'INAKTIV (Szenario A)'}"
        )

        offgrid_df = identify_offgrid_counties(
            shapes,
            bus_geom,
            bus_load_mw,
            bus_max_line_loading,
            elec_data,
            national_avg,
            cfg,
        )
    logger.info(f"  → {len(offgrid_df)} Offgrid-Counties identifiziert")

    if "kwh_per_person_yr" not in offgrid_df.columns:
        offgrid_df = offgrid_df.copy()
        offgrid_df["kwh_per_person_yr"] = float(cfg["kwh_per_person_yr"])

    if len(offgrid_df) == 0:
        logger.warning("Keine Offgrid-Counties! Netzwerk unveraendert speichern.")
        os.makedirs(os.path.dirname(snakemake.output.network), exist_ok=True)
        os.makedirs(os.path.dirname(snakemake.output.csv), exist_ok=True)
        n.export_to_netcdf(snakemake.output.network)
        pd.DataFrame().to_csv(snakemake.output.csv, index=False)
        exit(0)

    # ── Schritt 6: Reines Offgrid-Optimierungsnetz erzeugen ──────────
    n_base = n
    n_off = create_offgrid_optimization_network(n_base)

    # ── Schritt 7: Mini-Grid Busse hinzufügen ─────────────────────────
    logger.info(f"Fuege {len(offgrid_df)} Mini-Grid Busse ein...")
    n_added = 0

    for _, row in offgrid_df.iterrows():
        rid = row["gadm_id"]
        result = add_offgrid_bus(
            n=n_off,
            region_id=rid,
            population=row["population"],
            kwh_per_person_yr=row["kwh_per_person_yr"],
            centroid_x=row["centroid_x"],
            centroid_y=row["centroid_y"],
            cfg=cfg,
            cutout=cutout,
            annuity_factor=annuity_factor,
            snapshots=snapshots,
            snapshot_weightings=snapshot_weightings,
            offgrid_costs=offgrid_costs,
        )
        if result is not None:
            n_added += 1

    logger.info(f"  → {n_added} Busse hinzugefuegt")
    logger.info(
        f"  → Offgrid-Optimierungsnetz: {len(n_off.buses)} Busse "
        f"({n_added} Offgrid, Hauptnetz bleibt unveraendert)"
    )

    # ── Schritt 8: Offgrid-only optimieren ───────────────────────────
    solver_options = cfg.get("solver_options", {})
    logger.info(f"Offgrid-Optimierung: {cfg['solver']} | {len(n_off.buses)} Busse")
    if solver_options:
        logger.info(f"Solver-Optionen: {solver_options}")
    n_off.optimize(solver_name=cfg["solver"], solver_options=solver_options)

    # ── Schritt 9: Ergebnisse ────────────────────────────────────────
    results_df = extract_results(
        n_off, offgrid_df, cfg, annuity_factor, snapshot_weightings, offgrid_costs
    )

    if len(results_df) > 0:
        os.makedirs(os.path.dirname(snakemake.output.csv), exist_ok=True)
        results_df.to_csv(snakemake.output.csv, index=False)

        logger.info("=" * 60)
        logger.info("ZUSAMMENFASSUNG")
        logger.info("=" * 60)
        logger.info(f"  Mini-Grids:        {len(results_df)}")
        logger.info(f"  Gesamtbevölkerung: {results_df['population'].sum():,}")
        logger.info(f"  Gesamt Solar:      {results_df['solar_kw'].sum():.0f} kW")
        logger.info(
            f"  Ø LCOE:            {results_df['lcoe_eur_kwh'].mean():.3f} EUR/kWh"
        )
        logger.info(f"  Ø Autarkie:        {results_df['autarky_pct'].mean():.1f}%")
        logger.info(f"  Gesamt CO2:        {results_df['co2_t_yr'].sum():.1f} t/Jahr")
        logger.info(f"  Ergebnisse:        {snakemake.output.csv}")

    # ── Schritt 10: Offgrid-Ergebnisse ins unveraenderte Basisnetz mergen ─
    n_out = merge_offgrid_results_into_base(n_base, n_off)
    os.makedirs(os.path.dirname(snakemake.output.network), exist_ok=True)
    n_out.export_to_netcdf(snakemake.output.network)
    logger.info(f"Finales Netzwerk: {snakemake.output.network}")
    logger.info("FERTIG ✅")
