# SPDX-FileCopyrightText: 2026 Timon Geiss
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Shared techno-economic assumptions for the H2G offgrid modules."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

try:
    from _helpers import (
        apply_currency_conversion,
        build_currency_conversion_cache,
        read_csv_nafix,
    )
except ModuleNotFoundError:  # pragma: no cover - direct import from repo root
    from scripts._helpers import (
        apply_currency_conversion,
        build_currency_conversion_cache,
        read_csv_nafix,
    )

logger = logging.getLogger(__name__)


def _annuity(lifetime: pd.Series, discount_rate: pd.Series) -> pd.Series:
    return discount_rate / (1.0 - 1.0 / (1.0 + discount_rate) ** lifetime)


def _normalise_token(value: Any) -> str:
    return "".join(char for char in str(value).strip().lower() if char.isalnum())


def _positive_float(value: Any, name: str) -> float:
    value = pd.to_numeric(value, errors="coerce")
    if pd.isna(value) or not np.isfinite(float(value)):
        raise ValueError(f"Invalid PyPSA cost value for {name}: {value}")
    return float(value)


def _scenario_resource_root(path: str | os.PathLike[str]) -> Path | None:
    parts = Path(path).parts
    if "resources" not in parts:
        return None
    idx = parts.index("resources")
    if idx + 1 >= len(parts):
        return None
    return Path(*parts[: idx + 2])


def resolve_tech_costs_path(snakemake: Any, full_config: dict[str, Any]) -> Path:
    """Resolve the scenario-specific PyPSA-Earth technology cost file."""
    input_obj = getattr(snakemake, "input", None)
    candidate = getattr(input_obj, "tech_costs", None)
    if candidate:
        candidate = Path(candidate)
        if candidate.exists():
            return candidate

    shapes = getattr(input_obj, "shapes", None)
    if shapes:
        root = _scenario_resource_root(shapes)
        if root is not None:
            year = full_config.get("costs", {}).get("year")
            if year is not None:
                candidate = root / f"costs_{year}.csv"
                if candidate.exists():
                    return candidate

    year = full_config.get("costs", {}).get("year")
    if year is not None:
        for candidate in sorted(Path("resources").glob(f"*/costs_{year}.csv")):
            if candidate.exists():
                return candidate

    raise FileNotFoundError(
        "Could not resolve PyPSA technology cost file. Expected snakemake.input.tech_costs "
        "or a scenario resources/<name>/costs_<year>.csv file."
    )


def _scenario_name_from_path(path: str | os.PathLike[str]) -> str | None:
    parts = Path(path).parts
    for part in parts:
        if part.startswith("H2G_A_CD_"):
            return part
    return None


def _ensure_full_config(snakemake: Any, full_config: dict[str, Any]) -> dict[str, Any]:
    if "costs" in full_config and "electricity" in full_config:
        return full_config

    network_path = getattr(getattr(snakemake, "input", None), "network", "")
    scenario_name = _scenario_name_from_path(network_path)
    if scenario_name is None:
        raise ValueError(
            "Offgrid PyPSA cost derivation needs full scenario config with 'costs' "
            "and 'electricity'. Could not infer scenario config from network path."
        )

    config_path = Path("configs/scenarios_H2G") / f"config.{scenario_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            "Offgrid PyPSA cost derivation needs full scenario config. "
            f"Expected {config_path}."
        )

    with open(config_path) as fh:
        loaded = yaml.safe_load(fh)
    merged = dict(loaded)
    merged.update(full_config)
    if "offgrid" in full_config:
        merged["offgrid"] = full_config["offgrid"]
    return merged


def load_pypsa_costs(
    tech_costs: str | os.PathLike[str],
    cost_config: dict[str, Any],
    elec_config: dict[str, Any],
    n_years: float = 1.0,
) -> pd.DataFrame:
    """Load the subset of PyPSA-Earth cost logic needed by the offgrid module."""
    costs = read_csv_nafix(tech_costs, index_col=["technology", "parameter"]).sort_index()

    costs.loc[costs.unit.str.contains("/kW", na=False), "value"] *= 1e3
    costs.unit = costs.unit.str.replace("/kW", "/MW", regex=False)
    cache = build_currency_conversion_cache(
        costs,
        output_currency=cost_config["output_currency"],
        default_exchange_rate=cost_config["default_exchange_rate"],
        future_exchange_rate_strategy=cost_config.get("future_exchange_rate_strategy"),
        custom_future_exchange_rate=cost_config.get("custom_future_exchange_rate"),
    )
    costs = apply_currency_conversion(costs, cost_config["output_currency"], cache)

    for col in ["scenario", "financial_case"]:
        if col in costs.columns:
            costs[col] = costs[col].replace("", pd.NA)

    if "scenario" in costs.columns:
        scenario = str(cost_config["cost_scenario"]).casefold()
        costs = costs[
            costs["scenario"].str.casefold().eq(scenario) | costs["scenario"].isnull()
        ]

    if "financial_case" in costs.columns:
        financial_case = str(cost_config["financial_case"]).casefold()
        costs = costs[
            costs["financial_case"].str.casefold().eq(financial_case)
            | costs["financial_case"].isnull()
        ]

    costs = costs.value.unstack().fillna(cost_config["fill_values"])

    for attr in ("investment", "lifetime", "FOM", "VOM", "efficiency", "fuel"):
        overwrites = cost_config.get(attr)
        if overwrites is None:
            continue
        overwrites = pd.Series(overwrites)
        known = overwrites.index.intersection(costs.index)
        if len(known):
            costs.loc[known, attr] = overwrites.loc[known]
        missing = overwrites.index.difference(costs.index)
        if len(missing):
            logger.warning(
                "Ignoring cost overwrite for unknown technologies: %s",
                ", ".join(map(str, missing)),
            )

    costs["capital_cost"] = (
        (_annuity(costs["lifetime"], costs["discount rate"]) + costs["FOM"] / 100.0)
        * costs["investment"]
        * float(n_years)
    )

    if "gas" in costs.index:
        for tech in ["OCGT", "CCGT"]:
            if tech in costs.index:
                costs.at[tech, "fuel"] = costs.at["gas", "fuel"]

    costs["marginal_cost"] = costs["VOM"] + costs["fuel"] / costs["efficiency"]
    costs = costs.rename(columns={"CO2 intensity": "co2_emissions"})

    if "gas" in costs.index:
        for tech in ["OCGT", "CCGT"]:
            if tech in costs.index:
                costs.at[tech, "co2_emissions"] = costs.at["gas", "co2_emissions"]

    if {"solar-rooftop", "solar-utility"}.issubset(costs.index):
        rooftop_share = float(cost_config.get("rooftop_share", 0.0))
        costs.at["solar", "capital_cost"] = (
            rooftop_share * costs.at["solar-rooftop", "capital_cost"]
            + (1.0 - rooftop_share) * costs.at["solar-utility", "capital_cost"]
        )

    max_hours = elec_config.get("max_hours", {})
    if {"battery storage", "battery inverter"}.issubset(costs.index):
        battery_max_hours = float(max_hours.get("battery", 6.0))
        costs.loc["battery", ["capital_cost", "marginal_cost", "co2_emissions"]] = [
            costs.at["battery inverter", "capital_cost"]
            + battery_max_hours * costs.at["battery storage", "capital_cost"],
            0.0,
            0.0,
        ]

    for attr in ("marginal_cost", "capital_cost"):
        overwrites = cost_config.get(attr)
        if overwrites is None:
            continue
        overwrites = pd.Series(overwrites)
        known = overwrites.index.intersection(costs.index)
        if len(known):
            costs.loc[known, attr] = overwrites.loc[known]

    return costs


def _network_load_shedding_cost(network: Any) -> float | None:
    generators = getattr(network, "generators", pd.DataFrame())
    if generators.empty or "carrier" not in generators.columns:
        return None
    carrier = generators["carrier"].map(_normalise_token)
    mask = carrier.eq("loadshedding")
    if not mask.any() or "marginal_cost" not in generators.columns:
        return None
    values = pd.to_numeric(generators.loc[mask, "marginal_cost"], errors="coerce")
    values = values[np.isfinite(values)]
    values = values[values > 0.0]
    if values.empty:
        return None
    return float(values.median())


def _config_load_shedding_cost(full_config: dict[str, Any]) -> float | None:
    value = full_config.get("solving", {}).get("options", {}).get("load_shedding")
    if value is None or value is False:
        return None
    return float(value) * 1000.0


def build_offgrid_costs(
    network: Any,
    snakemake: Any,
    full_config: dict[str, Any],
) -> dict[str, float]:
    """Build offgrid component costs from PyPSA-Earth project assumptions."""
    full_config = _ensure_full_config(snakemake, full_config)
    tech_costs = resolve_tech_costs_path(snakemake, full_config)
    n_years = 1.0
    weights = getattr(network, "snapshot_weightings", None)
    if isinstance(weights, pd.DataFrame) and "objective" in weights.columns:
        n_years = float(weights["objective"].sum()) / 8760.0
    elif isinstance(weights, pd.Series):
        n_years = float(weights.sum()) / 8760.0
    if not np.isfinite(n_years) or n_years <= 0.0:
        n_years = 1.0

    costs = load_pypsa_costs(
        tech_costs=tech_costs,
        cost_config=full_config["costs"],
        elec_config=full_config["electricity"],
        n_years=n_years,
    )

    required = ["solar", "battery", "battery inverter", "oil"]
    missing = [technology for technology in required if technology not in costs.index]
    if missing:
        raise ValueError(
            "Missing required PyPSA cost technologies for offgrid module: "
            + ", ".join(missing)
        )

    load_shedding_cost = _network_load_shedding_cost(network)
    if load_shedding_cost is None:
        load_shedding_cost = _config_load_shedding_cost(full_config)
    if load_shedding_cost is None:
        raise ValueError(
            "Could not derive PyPSA load-shedding marginal cost from network or config."
        )

    oil_efficiency = _positive_float(costs.at["oil", "efficiency"], "oil efficiency")
    oil_co2 = _positive_float(costs.at["oil", "co2_emissions"], "oil co2_emissions")
    battery_efficiency = _positive_float(
        costs.at["battery inverter", "efficiency"],
        "battery inverter efficiency",
    )

    offgrid_costs = {
        "solar_capital_cost_eur_per_mw_a": _positive_float(
            costs.at["solar", "capital_cost"], "solar capital_cost"
        ),
        "solar_marginal_cost_eur_per_mwh": _positive_float(
            costs.at["solar", "marginal_cost"], "solar marginal_cost"
        ),
        "battery_capital_cost_eur_per_mw_a": _positive_float(
            costs.at["battery", "capital_cost"], "battery capital_cost"
        ),
        "battery_marginal_cost_eur_per_mwh": _positive_float(
            costs.at["battery", "marginal_cost"], "battery marginal_cost"
        ),
        "battery_max_hours": float(
            full_config["electricity"].get("max_hours", {}).get("battery", 6.0)
        ),
        "battery_efficiency_store": battery_efficiency,
        "battery_efficiency_dispatch": battery_efficiency,
        "diesel_capital_cost_eur_per_mw_a": _positive_float(
            costs.at["oil", "capital_cost"], "oil capital_cost"
        ),
        "diesel_marginal_cost_eur_per_mwh": _positive_float(
            costs.at["oil", "marginal_cost"], "oil marginal_cost"
        ),
        "diesel_efficiency": oil_efficiency,
        "diesel_co2_t_per_mwh_el": oil_co2 / oil_efficiency,
        "load_shedding_marginal_cost_eur_per_mwh": float(load_shedding_cost),
    }

    logger.info(
        "Offgrid PyPSA costs from %s: solar %.2f EUR/MW/a, battery %.2f EUR/MW/a, "
        "diesel/oil %.2f EUR/MW/a, load shedding %.2f EUR/MWh.",
        tech_costs,
        offgrid_costs["solar_capital_cost_eur_per_mw_a"],
        offgrid_costs["battery_capital_cost_eur_per_mw_a"],
        offgrid_costs["diesel_capital_cost_eur_per_mw_a"],
        offgrid_costs["load_shedding_marginal_cost_eur_per_mwh"],
    )
    return offgrid_costs
