# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText:  PyPSA-Earth and PyPSA-Eur Authors
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# -*- coding: utf-8 -*-
"""
Adds electrical generators, load and existing hydro storage units to a base
network.

Relevant Settings
-----------------

.. code:: yaml

    costs:
        year:
        technology_data_version:
        discountrate:
        output_currency:
        country_specific_data:
        cost_scenario:
        financial_case:
        output_currency:
        default_exchange_rate:
        future_exchange_rate_strategy:
        custom_future_exchange_rate:
        rooftop_share:
        emission_prices:

    electricity:
        max_hours:
        marginal_cost:
        capital_cost:
        conventional_carriers:
        co2limit:
        extendable_carriers:
        include_renewable_capacities_from_OPSD:
        estimate_renewable_capacities_from_capacity_stats:

    renewable:
        hydro:
            carriers:
            hydro_max_hours:
            hydro_max_hours_default:
            hydro_capital_cost:

    lines:
        length_factor:

.. seealso::
    Documentation of the configuration file ``config.yaml`` at :ref:`costs_cf`,
    :ref:`electricity_cf`, :ref:`load_options_cf`, :ref:`renewable_cf`, :ref:`lines_cf`

Inputs
------

- ``resources/costs.csv``: The database of cost assumptions for all included technologies for specific years from various sources; e.g. discount rate, lifetime, investment (CAPEX), fixed operation and maintenance (FOM), variable operation and maintenance (VOM), fuel costs, efficiency, carbon-dioxide intensity.
- ``data/bundle/hydro_capacities.csv``: Hydropower plant store/discharge power capacities, energy storage capacity, and average hourly inflow by country.  Not currently used!

    .. image:: /img/hydrocapacities.png

- ``data/geth2015_hydro_capacities.csv``: alternative to capacities above; not currently used!
- ``resources/demand_profiles.csv``: a csv file containing the demand profile associated with buses
- ``resources/shapes/gadm_shapes.geojson``: confer :ref:`shapes`
- ``resources/powerplants.csv``: confer :ref:`powerplants`
- ``resources/profile_{}.nc``: all technologies in ``config["renewables"].keys()``, confer :ref:`renewableprofiles`
- ``networks/base.nc``: confer :ref:`base`

Outputs
-------

- ``networks/elec.nc``:

    .. image:: /img/elec.png
            :width: 75 %
            :align: center

Description
-----------

The rule :mod:`add_electricity` ties all the different data inputs from the preceding rules together into a detailed PyPSA network that is stored in ``networks/elec.nc``. It includes:

- today's transmission topology and transfer capacities (in future, optionally including lines which are under construction according to the config settings ``lines: under_construction`` and ``links: under_construction``),
- today's thermal and hydro power generation capacities (for the technologies listed in the config setting ``electricity: conventional_carriers``), and
- today's load time-series (upsampled in a top-down approach according to population and gross domestic product)

It further adds extendable ``generators`` with **zero** capacity for

- photovoltaic, onshore and AC- as well as DC-connected offshore wind installations with today's locational, hourly wind and solar capacity factors (but **no** current capacities),
- additional open- and combined-cycle gas turbines (if ``OCGT`` and/or ``CCGT`` is listed in the config setting ``electricity: extendable_carriers``)
"""

import numpy as np
import pandas as pd
import powerplantmatching as pm
import pypsa
import xarray as xr
from _helpers import (
    apply_currency_conversion,
    build_currency_conversion_cache,
    configure_logging,
    create_logger,
    read_csv_nafix,
    sanitize_carriers,
    sanitize_locations,
    update_p_nom_max,
)
from powerplantmatching.export import map_country_bus

idx = pd.IndexSlice

logger = create_logger(__name__)


def normed(s):
    return s / s.sum()


def calculate_annuity(n, r):
    """
    Calculate the annuity factor for an asset with lifetime n years and
    discount rate of r, e.g. annuity(20, 0.05) * 20 = 1.6.
    """
    if isinstance(r, pd.Series):
        return pd.Series(1 / n, index=r.index).where(
            r == 0, r / (1.0 - 1.0 / (1.0 + r) ** n)
        )
    elif r > 0:
        return r / (1.0 - 1.0 / (1.0 + r) ** n)
    else:
        return 1 / n


def _add_missing_carriers_from_costs(n, costs, carriers):
    missing_carriers = pd.Index(carriers).difference(n.carriers.index)
    if missing_carriers.empty:
        return

    emissions_cols = (
        costs.columns.to_series().loc[lambda s: s.str.endswith("_emissions")].values
    )
    suptechs = missing_carriers.str.split("-").str[0]
    emissions = costs.loc[suptechs, emissions_cols].fillna(0.0)
    emissions.index = missing_carriers
    n.import_components_from_dataframe(emissions, "Carrier")


def load_costs(tech_costs, config, elec_config, Nyears=1):
    """
    Set all asset costs and other parameters.
    """
    costs = read_csv_nafix(tech_costs, index_col=["technology", "parameter"]).sort_index()

    # correct units to MW and output_currency
    costs.loc[costs.unit.str.contains("/kW"), "value"] *= 1e3
    costs.unit = costs.unit.str.replace("/kW", "/MW")
    _currency_conversion_cache = build_currency_conversion_cache(
        costs,
        output_currency=config["output_currency"],
        default_exchange_rate=config["default_exchange_rate"],
        future_exchange_rate_strategy=config.get("future_exchange_rate_strategy"),
        custom_future_exchange_rate=config.get("custom_future_rate", None),
    )
    costs = apply_currency_conversion(
        costs,
        config["output_currency"],
        _currency_conversion_cache,
    )

    # apply filter on financial_case and scenario, if they are contained in the cost dataframe
    wished_cost_scenario = config["cost_scenario"]
    wished_financial_case = config["financial_case"]
    for col in ["scenario", "financial_case"]:
        if col in costs.columns:
            costs[col] = costs[col].replace("", pd.NA)

    if "scenario" in costs.columns:
        costs = costs[
            (costs["scenario"].str.casefold() == wished_cost_scenario.casefold())
            | (costs["scenario"].isnull())
        ]

    if "financial_case" in costs.columns:
        costs = costs[
            (costs["financial_case"].str.casefold() == wished_financial_case.casefold())
            | (costs["financial_case"].isnull())
        ]

    costs = costs.value.unstack().fillna(config["fill_values"])

    for attr in ("investment", "lifetime", "FOM", "VOM", "efficiency", "fuel"):
        overwrites = config.get(attr)
        if overwrites is not None:
            overwrites = pd.Series(overwrites)
            costs.loc[overwrites.index, attr] = overwrites
            logger.info(
                f"Overwriting {attr} of {overwrites.index} to {overwrites.values}"
            )

    costs["capital_cost"] = (
        (
            calculate_annuity(costs["lifetime"], costs["discount rate"])
            + costs["FOM"] / 100.0
        )
        * costs["investment"]
        * Nyears
    )

    costs.at["OCGT", "fuel"] = costs.at["gas", "fuel"]
    costs.at["CCGT", "fuel"] = costs.at["gas", "fuel"]

    costs["marginal_cost"] = costs["VOM"] + costs["fuel"] / costs["efficiency"]

    costs = costs.rename(columns={"CO2 intensity": "co2_emissions"})
    # rename because technology data & pypsa earth costs.csv use different names
    # TODO: rename the technologies in hosted tutorial data to match technology data
    costs = costs.rename(
        {
            "hydrogen storage": "hydrogen storage tank",
            "hydrogen storage tank": "hydrogen storage tank",
            "hydrogen storage tank type 1": "hydrogen storage tank",
            "hydrogen underground storage": "hydrogen storage underground",
        },
    )

    costs.at["OCGT", "co2_emissions"] = costs.at["gas", "co2_emissions"]
    costs.at["CCGT", "co2_emissions"] = costs.at["gas", "co2_emissions"]

    costs.at["solar", "capital_cost"] = (
        config["rooftop_share"] * costs.at["solar-rooftop", "capital_cost"]
        + (1 - config["rooftop_share"]) * costs.at["solar-utility", "capital_cost"]
    )
    costs.loc["csp"] = costs.loc["csp-tower"]

    def costs_for_storage(store, link1, link2=None, max_hours=1.0):
        capital_cost = link1["capital_cost"] + max_hours * store["capital_cost"]
        if link2 is not None:
            capital_cost += link2["capital_cost"]
        return pd.Series(
            dict(capital_cost=capital_cost, marginal_cost=0.0, co2_emissions=0.0)
        )

    max_hours = elec_config["max_hours"]
    costs.loc["battery"] = costs_for_storage(
        costs.loc["battery storage"],
        costs.loc["battery inverter"],
        max_hours=max_hours["battery"],
    )
    costs.loc["H2"] = costs_for_storage(
        costs.loc["hydrogen storage tank"],
        costs.loc["fuel cell"],
        costs.loc["electrolysis"],
        max_hours=max_hours["H2"],
    )

    for attr in ("marginal_cost", "capital_cost"):
        overwrites = config.get(attr)
        if overwrites is not None:
            overwrites = pd.Series(overwrites)
            costs.loc[overwrites.index, attr] = overwrites
            logger.info(
                f"Overwriting {attr} of {overwrites.index} to {overwrites.values}"
            )

    return costs


def load_powerplants(ppl_fn):
    carrier_dict = {
        "ocgt": "OCGT",
        "ccgt": "CCGT",
        "bioenergy": "biomass",
        "ccgt, thermal": "CCGT",
        "hard coal": "coal",
    }
    ppl = (
        read_csv_nafix(ppl_fn, index_col=0, dtype={"bus": "str"})
        .powerplant.to_pypsa_names()
        .powerplant.convert_country_to_alpha2()
        .rename(columns=str.lower)
        .drop(columns=["efficiency"])
        .replace({"carrier": carrier_dict})
    )
    # drop powerplants with null capacity
    null_ppls = ppl[ppl.p_nom <= 0]
    if not null_ppls.empty:
        logger.warning(f"Drop powerplants with null capacity: {list(null_ppls.name)}.")
        ppl = ppl.drop(null_ppls.index)
    return ppl


def get_powerplant_candidate_config(electricity_config):
    config = electricity_config.get("custom_powerplant_candidates", {}) or {}
    if not config.get("enabled", False):
        return {"enabled": False}

    candidate_years = [int(year) for year in config.get("candidate_datein", [])]
    if not candidate_years:
        raise ValueError(
            "custom_powerplant_candidates.enabled is true, but candidate_datein is empty."
        )

    return {
        "enabled": True,
        "model_year": int(config["model_year"]),
        "existing_cutoff_year": int(config["existing_cutoff_year"]),
        "candidate_datein": candidate_years,
        "exclude_carriers": set(config.get("exclude_carriers", [])),
    }


def get_powerplant_candidate_mask(ppl, candidate_config, label):
    if not candidate_config.get("enabled", False) or ppl.empty:
        return pd.Series(False, index=ppl.index)

    datein = pd.to_numeric(
        ppl.get("datein", pd.Series(np.nan, index=ppl.index)), errors="coerce"
    )
    model_year = candidate_config["model_year"]
    cutoff_year = candidate_config["existing_cutoff_year"]
    candidate_years = candidate_config["candidate_datein"]
    exclude_carriers = candidate_config.get("exclude_carriers", set())

    if exclude_carriers:
        carrier = ppl.get("carrier", pd.Series("", index=ppl.index)).astype(str)
        excluded = carrier.isin(exclude_carriers)
    else:
        excluded = pd.Series(False, index=ppl.index)

    future = datein.notna() & (datein > cutoff_year)
    candidate_relevant = future & ~excluded
    too_late = candidate_relevant & (datein > model_year)
    unsupported = (
        candidate_relevant & (datein <= model_year) & ~datein.isin(candidate_years)
    )
    invalid = too_late | unsupported

    if invalid.any():
        cols = [col for col in ["name", "carrier", "technology", "datein"] if col in ppl]
        bad = ppl.loc[invalid, cols].head(10).to_dict("records")
        raise ValueError(
            f"Unexpected future {label} powerplants for model year {model_year}. "
            f"Future assets must have DateIn in {candidate_years} and DateIn <= model_year. "
            f"Examples: {bad}"
        )

    return candidate_relevant & datein.isin(candidate_years) & (datein <= model_year)


def get_existing_capacity_derating_config(electricity_config):
    config = electricity_config.get("existing_capacity_derating", {}) or {}
    if not config.get("enabled", False):
        return {"enabled": False}

    factor = float(config["factor"])
    if factor < 0 or factor > 1:
        raise ValueError(
            "existing_capacity_derating.factor must be between 0 and 1."
        )

    return {
        "enabled": True,
        "factor": factor,
        "existing_cutoff_year": int(config["existing_cutoff_year"]),
        "carriers": set(config.get("carriers", [])),
    }


def apply_existing_capacity_derating(
    ppl, p_nom, candidate, derating_config, label
):
    if not derating_config.get("enabled", False) or ppl.empty:
        return p_nom

    datein = pd.to_numeric(
        ppl.get("datein", pd.Series(np.nan, index=ppl.index)), errors="coerce"
    )
    existing = (
        (datein.isna() | (datein <= derating_config["existing_cutoff_year"]))
        & ~candidate
    )
    if derating_config.get("carriers"):
        carrier = ppl.get("carrier", pd.Series("", index=ppl.index))
        existing &= carrier.isin(derating_config["carriers"])

    if existing.any():
        factor = derating_config["factor"]
        logger.info(
            "Applying {:.0%} capacity derating to {} existing {} powerplants "
            "with {:.2f} MW original capacity.".format(
                1 - factor, existing.sum(), label, p_nom.loc[existing].sum()
            )
        )
        p_nom = p_nom.copy()
        p_nom.loc[existing] *= factor

    return p_nom


def apply_powerplant_candidate_nominal_attrs(
    ppl, candidate_config, derating_config, label
):
    candidate = get_powerplant_candidate_mask(ppl, candidate_config, label)
    derated_p_nom = apply_existing_capacity_derating(
        ppl, ppl["p_nom"], candidate, derating_config, label
    )
    if not candidate_config.get("enabled", False):
        return {
            "candidate": candidate,
            "p_nom": derated_p_nom,
        }

    return {
        "candidate": candidate,
        "p_nom": derated_p_nom.mask(candidate, 0.0),
        "p_nom_min": derated_p_nom.where(~candidate, 0.0),
        "p_nom_max": ppl["p_nom"].where(candidate, np.inf),
        "p_nom_extendable": candidate,
    }


def get_hydro_project_energy_attrs(ppl, candidate, label):
    annual = pd.to_numeric(
        ppl.get("annualgeneration_gwh", pd.Series(np.nan, index=ppl.index)),
        errors="coerce",
    )
    flh = pd.to_numeric(
        ppl.get("fullloadhours_h", pd.Series(np.nan, index=ppl.index)),
        errors="coerce",
    )

    invalid = candidate & (
        annual.isna() | flh.isna() | (annual <= 0) | (flh <= 0) | (flh > 8760)
    )
    if invalid.any():
        cols = [
            col
            for col in [
                "name",
                "technology",
                "p_nom",
                "datein",
                "annualgeneration_gwh",
                "fullloadhours_h",
            ]
            if col in ppl
        ]
        bad = ppl.loc[invalid, cols].to_dict("records")
        raise ValueError(
            f"Future {label} candidates require valid AnnualGeneration_GWh and "
            "FullLoadHours_h values, with 0 < FullLoadHours_h <= 8760. "
            f"Examples: {bad}"
        )

    return annual.where(candidate), flh.where(candidate), ppl["p_nom"].where(candidate)


def scale_hydro_candidate_inflow_timeseries(
    inflow_t, plants, candidate, label, cap_at_p_nom=False
):
    """
    Use existing hydro inflow as relative shape and rescale future candidates
    to project-specific annual generation.
    """
    if not candidate.any():
        return inflow_t

    candidate_plants = plants.loc[candidate].copy()
    candidate_cols = candidate_plants.index.intersection(inflow_t.columns)
    missing_cols = candidate_plants.index.difference(candidate_cols)
    if not missing_cols.empty:
        raise ValueError(
            f"Future {label} candidates are missing hydro inflow profiles: "
            f"{', '.join(map(str, missing_cols))}."
        )

    annual_generation_gwh = pd.to_numeric(
        candidate_plants.loc[candidate_cols, "annualgeneration_gwh"], errors="coerce"
    )
    full_load_hours = pd.to_numeric(
        candidate_plants.loc[candidate_cols, "fullloadhours_h"], errors="coerce"
    )
    capacity = pd.to_numeric(
        candidate_plants.loc[candidate_cols, "p_nom"], errors="coerce"
    )
    invalid = (
        annual_generation_gwh.isna()
        | full_load_hours.isna()
        | capacity.isna()
        | (annual_generation_gwh <= 0)
        | (full_load_hours <= 0)
        | (full_load_hours > 8760)
        | (capacity <= 0)
    )
    if invalid.any():
        cols = [
            col
            for col in [
                "name",
                "technology",
                "p_nom",
                "datein",
                "annualgeneration_gwh",
                "fullloadhours_h",
            ]
            if col in candidate_plants
        ]
        bad = candidate_plants.loc[candidate_cols[invalid], cols].to_dict("records")
        raise ValueError(
            f"Invalid project-specific energy data for future {label} candidates. "
            "AnnualGeneration_GWh, FullLoadHours_h and Capacity must be positive, "
            f"and FullLoadHours_h must be <= 8760. Examples: {bad}"
        )

    selected = inflow_t.loc[:, candidate_cols]
    original_mean = selected.mean(axis=0)
    zero_mean = original_mean <= 0
    if zero_mean.any():
        bad = original_mean.index[zero_mean]
        raise ValueError(
            f"Future {label} candidates have zero-mean hydro inflow profiles and "
            f"cannot be rescaled: {', '.join(map(str, bad))}."
        )

    relative = selected.divide(original_mean, axis=1)
    target_mean = annual_generation_gwh * 1e3 / len(inflow_t.index)
    scaled = relative.multiply(target_mean, axis=1)

    if cap_at_p_nom:
        for plant in candidate_cols:
            shape = relative[plant].fillna(0.0)
            plant_capacity = float(capacity.loc[plant])
            target_mwh = float(annual_generation_gwh.loc[plant] * 1e3)
            max_mwh = plant_capacity * len(shape)

            if target_mwh >= max_mwh:
                logger.warning(
                    "%s candidate %s requests %.2f GWh/a, which reaches or "
                    "exceeds the p_max_pu <= 1 limit of %.2f GWh/a. Capping "
                    "at full output.",
                    label,
                    plant,
                    target_mwh / 1e3,
                    max_mwh / 1e3,
                )
                scaled[plant] = plant_capacity
                continue

            low = 0.0
            high = float(target_mean.loc[plant])
            while np.minimum(shape * high, plant_capacity).sum() < target_mwh:
                high *= 2.0

            for _ in range(60):
                mid = (low + high) / 2.0
                annual_mwh = np.minimum(shape * mid, plant_capacity).sum()
                if annual_mwh < target_mwh:
                    low = mid
                else:
                    high = mid

            scaled[plant] = np.minimum(shape * high, plant_capacity)

    result = inflow_t.copy()
    result.loc[:, candidate_cols] = scaled
    logger.info(
        "Scaled %d future %s inflow profiles to project-specific annual "
        "generation data (%.2f TWh/a target, %.2f TWh/a profile).",
        len(candidate_cols),
        label,
        annual_generation_gwh.sum() / 1e3,
        result.loc[:, candidate_cols].sum().sum() / 1e6,
    )
    return result


def attach_load(n, demand_profiles):
    """
    Add load profiles to network buses.

    Parameters
    ----------
    n: pypsa network

    demand_profiles: str
        Path to csv file of elecric demand time series, e.g. "resources/demand_profiles.csv"
        Demand profile has snapshots as rows and bus names as columns.

    Returns
    -------
    n : pypsa network
        Now attached with load time series
    """
    demand_df = read_csv_nafix(demand_profiles, index_col=0, parse_dates=True)

    n.madd("Load", demand_df.columns, bus=demand_df.columns, p_set=demand_df)


def attach_dc_costs(lines_or_links, costs, length_factor=1.0, simple_hvdc_costs=False):
    if lines_or_links.empty:
        return

    if lines_or_links.loc[lines_or_links.carrier == "DC"].empty:
        return

    dc_b = lines_or_links.carrier == "DC"
    if simple_hvdc_costs:
        costs = (
            lines_or_links.loc[dc_b, "length"]
            * length_factor
            * costs.at["HVDC overhead", "capital_cost"]
        )
    else:
        costs = (
            lines_or_links.loc[dc_b, "length"]
            * length_factor
            * (
                (1.0 - lines_or_links.loc[dc_b, "underwater_fraction"])
                * costs.at["HVDC overhead", "capital_cost"]
                + lines_or_links.loc[dc_b, "underwater_fraction"]
                * costs.at["HVDC submarine", "capital_cost"]
            )
            + costs.at["HVDC inverter pair", "capital_cost"]
        )
    lines_or_links.loc[dc_b, "capital_cost"] = costs


def update_transmission_costs(n, costs, length_factor=1.0, simple_hvdc_costs=False):
    n.lines["capital_cost"] = (
        n.lines["length"] * length_factor * costs.at["HVAC overhead", "capital_cost"]
    )

    attach_dc_costs(
        lines_or_links=n.links,
        costs=costs,
        length_factor=length_factor,
        simple_hvdc_costs=simple_hvdc_costs,
    )
    attach_dc_costs(
        lines_or_links=n.lines,
        costs=costs,
        length_factor=length_factor,
        simple_hvdc_costs=simple_hvdc_costs,
    )


def attach_wind_and_solar(
    n,
    costs,
    ppl,
    input_files,
    technologies,
    extendable_carriers,
    line_length_factor,
):
    # TODO: rename tech -> carrier, technologies -> carriers
    _add_missing_carriers_from_costs(n, costs, technologies)

    df = ppl.rename(columns={"country": "Country"})

    for tech in technologies:
        if tech == "hydro":
            continue

        if tech == "offwind-ac":
            # add all offwind wind power plants by default as offwind-ac
            df["carrier"] = df["carrier"].mask(
                df.technology == "Offshore", "offwind-ac"
            )

        df["carrier"] = df["carrier"].mask(df.technology == "Onshore", "onwind")

        with xr.open_dataset(getattr(input_files, "profile_" + tech)) as ds:
            if ds.indexes["bus"].empty:
                continue

            suptech = tech.split("-", 2)[0]
            if suptech == "offwind":
                underwater_fraction = ds["underwater_fraction"].to_pandas()
                connection_cost = (
                    line_length_factor
                    * ds["average_distance"].to_pandas()
                    * (
                        underwater_fraction
                        * costs.at[tech + "-connection-submarine", "capital_cost"]
                        + (1.0 - underwater_fraction)
                        * costs.at[tech + "-connection-underground", "capital_cost"]
                    )
                )
                capital_cost = (
                    costs.at["offwind", "capital_cost"]
                    + costs.at[tech + "-station", "capital_cost"]
                    + connection_cost
                )
                logger.info(
                    "Added connection cost of {:0.0f}-{:0.0f} {}/MW/a to {}".format(
                        connection_cost.min(),
                        connection_cost.max(),
                        snakemake.params.costs["output_currency"],
                        tech,
                    )
                )
            else:
                capital_cost = costs.at[tech, "capital_cost"]

            if not df.query("carrier == @tech").empty:
                buses = n.buses.loc[ds.indexes["bus"]]
                caps = map_country_bus(df.query("carrier == @tech"), buses)
                caps = caps.groupby(["bus"]).p_nom.sum()
                caps = pd.Series(data=caps, index=ds.indexes["bus"]).fillna(0)
            else:
                caps = pd.Series(index=ds.indexes["bus"]).fillna(0)

            n.madd(
                "Generator",
                ds.indexes["bus"],
                " " + tech,
                bus=ds.indexes["bus"],
                carrier=tech,
                p_nom=caps,
                p_nom_extendable=tech in extendable_carriers["Generator"],
                p_nom_min=caps,
                p_nom_max=ds["p_nom_max"].to_pandas(),
                p_max_pu=ds["profile"].transpose("time", "bus").to_pandas(),
                weight=ds["weight"].to_pandas(),
                marginal_cost=costs.at[suptech, "marginal_cost"],
                capital_cost=capital_cost,
                efficiency=costs.at[suptech, "efficiency"],
            )


def attach_conventional_generators(
    n,
    costs,
    ppl,
    conventional_carriers,
    extendable_carriers,
    renewable_carriers,
    conventional_config,
    extendable_existing_limits,
    conventional_inputs,
    candidate_config,
    derating_config,
):
    carriers = set(conventional_carriers) | (
        set(extendable_carriers["Generator"]) - set(renewable_carriers)
    )
    _add_missing_carriers_from_costs(n, costs, carriers)

    ppl = (
        ppl.query("carrier in @carriers")
        .join(costs, on="carrier", rsuffix="_r")
        .rename(index=lambda s: "C" + str(s))
    )
    ppl["efficiency"] = ppl.efficiency.fillna(ppl.efficiency)

    logger.info(
        "Adding {} generators with capacities [GW] \n{}".format(
            len(ppl), ppl.groupby("carrier").p_nom.sum().div(1e3).round(2)
        )
    )

    candidate = get_powerplant_candidate_mask(
        ppl, candidate_config, "conventional"
    )
    derated_p_nom = apply_existing_capacity_derating(
        ppl, ppl.p_nom, candidate, derating_config, "conventional"
    )
    if candidate_config.get("enabled", False):
        extendable_existing = ppl.carrier.isin(extendable_carriers["Generator"])
        p_nom_extendable = candidate | extendable_existing
        p_nom = derated_p_nom.mask(candidate, 0.0)
        p_nom_min = derated_p_nom.where(~candidate, 0.0)
        p_nom_max = pd.Series(np.inf, index=ppl.index).where(~candidate, ppl.p_nom)
        for carrier, limits in (extendable_existing_limits or {}).items():
            carrier_extendable = extendable_existing & ppl.carrier.eq(carrier)
            if not carrier_extendable.any():
                continue

            if "p_nom_min" in limits:
                p_nom_min.loc[carrier_extendable] = float(limits["p_nom_min"])
            if limits.get("p_nom_max") == "capacity":
                p_nom_max.loc[carrier_extendable] = ppl.loc[
                    carrier_extendable, "p_nom"
                ]
            elif "p_nom_max" in limits:
                p_nom_max.loc[carrier_extendable] = float(limits["p_nom_max"])

            invalid = carrier_extendable & (p_nom_min > p_nom_max)
            if invalid.any():
                bad = ppl.loc[invalid, ["name", "carrier", "p_nom"]].to_dict(
                    "records"
                )
                raise ValueError(
                    "Invalid extendable existing limits for conventional "
                    f"{carrier} generators. Examples: {bad}"
                )

        if candidate.any():
            logger.info(
                "Treating {} conventional generators as build candidates with {:.2f} MW p_nom_max.".format(
                    candidate.sum(), ppl.loc[candidate, "p_nom"].sum()
                )
            )
        if extendable_existing.any():
            logger.info(
                "Allowing {} existing conventional generators to expand beyond their fixed minimum capacity.".format(
                    extendable_existing.sum()
                )
            )
    else:
        p_nom_extendable = ppl.carrier.isin(extendable_carriers["Generator"])
        p_nom = derated_p_nom.where(ppl.carrier.isin(conventional_carriers), 0)
        p_nom_min = derated_p_nom.where(
            ppl.carrier.isin(conventional_carriers), 0
        )
        p_nom_max = pd.Series(np.inf, index=ppl.index)

    n.madd(
        "Generator",
        ppl.index,
        carrier=ppl.carrier,
        bus=ppl.bus,
        p_nom_min=p_nom_min,
        p_nom=p_nom,
        p_nom_max=p_nom_max,
        p_nom_extendable=p_nom_extendable,
        efficiency=ppl.efficiency,
        marginal_cost=ppl.marginal_cost,
        capital_cost=ppl.capital_cost,
        build_year=ppl.datein.fillna(0).astype(int),
        lifetime=(ppl.dateout - ppl.datein).fillna(np.inf),
    )

    for carrier in conventional_config:
        # Generators with technology affected
        idx = n.generators.query("carrier == @carrier").index

        for attr in list(set(conventional_config[carrier]) & set(n.generators)):
            values = conventional_config[carrier][attr]

            if f"conventional_{carrier}_{attr}" in conventional_inputs:
                # Values affecting generators of technology k country-specific
                # First map generator buses to countries; then map countries to p_max_pu
                values = read_csv_nafix(values, index_col=0).iloc[:, 0]
                bus_values = n.buses.country.map(values)
                n.generators[attr].update(
                    n.generators.loc[idx].bus.map(bus_values).dropna()
                )
            else:
                # Single value affecting all generators of technology k indiscriminantely of country
                n.generators.loc[idx, attr] = values


def attach_hydro(n, costs, ppl, candidate_config, derating_config):
    if "hydro" not in snakemake.params.renewable:
        return
    c = snakemake.params.renewable["hydro"]
    carriers = c.get("carriers", ["ror", "PHS", "hydro"])

    _add_missing_carriers_from_costs(n, costs, carriers)

    ppl = (
        ppl.query('carrier == "hydro"')
        .assign(ppl_id=lambda df: df.index)
        .reset_index(drop=True)
        .rename(index=lambda s: str(s) + " hydro")
    )

    # Current fix, NaN technologies set to ROR
    if ppl.technology.isna().any():
        n_nans = ppl.technology.isna().sum()
        logger.warning(
            f"Identified {n_nans} hydro powerplants with unknown technology.\n"
            "Initialized to 'Run-Of-River'"
        )
        ppl.loc[ppl.technology.isna(), "technology"] = "Run-Of-River"

    ror = ppl.query('technology == "Run-Of-River"')
    phs = ppl.query('technology == "Pumped Storage"')
    hydro = ppl.query('technology == "Reservoir"')

    inflow_idx = ror.index.union(hydro.index)
    if not inflow_idx.empty:
        with xr.open_dataarray(snakemake.input.profile_hydro) as inflow:
            found_plants = ppl.ppl_id[ppl.ppl_id.isin(inflow.indexes["plant"])]
            missing_plants_idxs = ppl.index.difference(found_plants.index)

            # if missing time series are found, notify the user and exclude missing hydro plants
            if not missing_plants_idxs.empty:
                # original total p_nom
                total_p_nom = ror.p_nom.sum() + hydro.p_nom.sum()

                ror = ror.loc[ror.index.intersection(found_plants.index)]
                hydro = hydro.loc[hydro.index.intersection(found_plants.index)]
                # loss of p_nom
                loss_p_nom = ror.p_nom.sum() + hydro.p_nom.sum() - total_p_nom

                logger.warning(
                    f"'{snakemake.input.profile_hydro}' is missing inflow time-series for at least one bus: {', '.join(missing_plants_idxs)}."
                    f"Corresponding hydro plants are dropped, corresponding to a total loss of {loss_p_nom:.2f}MW out of {total_p_nom:.2f}MW."
                )

            # if there are any plants for which runoff data are available
            if not found_plants.empty:
                inflow_t = (
                    inflow.sel(plant=found_plants.values)
                    .assign_coords(plant=found_plants.index)
                    .rename({"plant": "name"})
                    .transpose("time", "name")
                    .to_pandas()
                )

    if "ror" in carriers and not ror.empty:
        ror_nominal = apply_powerplant_candidate_nominal_attrs(
            ror, candidate_config, derating_config, "run-of-river hydro"
        )
        ror_candidate = ror_nominal.pop("candidate")
        if ror_candidate.any():
            logger.info(
                "Treating {} run-of-river hydro generators as build candidates with {:.2f} MW p_nom_max.".format(
                    ror_candidate.sum(), ror.loc[ror_candidate, "p_nom"].sum()
                )
            )
        (
            ror_annual_generation,
            ror_full_load_hours,
            ror_candidate_capacity,
        ) = get_hydro_project_energy_attrs(ror, ror_candidate, "run-of-river hydro")
        ror_inflow_t = inflow_t.loc[:, ror.index].copy()
        if ror_candidate.any():
            ror_inflow_t = scale_hydro_candidate_inflow_timeseries(
                ror_inflow_t,
                ror,
                ror_candidate,
                "run-of-river hydro",
                cap_at_p_nom=True,
            )
        n.madd(
            "Generator",
            ror.index,
            carrier="ror",
            bus=ror["bus"],
            **ror_nominal,
            efficiency=costs.at["ror", "efficiency"],
            capital_cost=costs.at["ror", "capital_cost"],
            weight=ror["p_nom"],
            p_max_pu=(
                ror_inflow_t
                .divide(ror["p_nom"], axis=1)
                .where(lambda df: df <= 1.0, other=1.0)
            ),
        )
        n.generators.loc[ror.index, "annualgeneration_gwh"] = ror_annual_generation
        n.generators.loc[ror.index, "fullloadhours_h"] = ror_full_load_hours
        n.generators.loc[ror.index, "candidate_capacity_mw"] = ror_candidate_capacity

    if "PHS" in carriers and not phs.empty:
        # fill missing max hours to config value and
        # assume no natural inflow due to lack of data
        phs = phs.replace({"max_hours": {0: c["PHS_max_hours"]}})
        phs_nominal = apply_powerplant_candidate_nominal_attrs(
            phs, candidate_config, derating_config, "pumped-storage hydro"
        )
        phs_candidate = phs_nominal.pop("candidate")
        if phs_candidate.any():
            logger.info(
                "Treating {} pumped-storage hydro units as build candidates with {:.2f} MW p_nom_max.".format(
                    phs_candidate.sum(), phs.loc[phs_candidate, "p_nom"].sum()
                )
            )
        n.madd(
            "StorageUnit",
            phs.index,
            carrier="PHS",
            bus=phs["bus"],
            **phs_nominal,
            capital_cost=costs.at["PHS", "capital_cost"],
            max_hours=phs["max_hours"],
            efficiency_store=np.sqrt(costs.at["PHS", "efficiency"]),
            efficiency_dispatch=np.sqrt(costs.at["PHS", "efficiency"]),
            cyclic_state_of_charge=True,
        )

    if "hydro" in carriers and not hydro.empty:
        hydro_max_hours = c.get("hydro_max_hours")
        hydro_stats = (
            read_csv_nafix(
                snakemake.input.hydro_capacities,
                comment="#",
                na_values=["-"],
                index_col=0,
            )
            .groupby("Country")
            .sum()
        )
        e_target = hydro_stats["E_store[TWh]"].clip(lower=0.2) * 1e6
        e_installed = hydro.eval("p_nom * max_hours").groupby(hydro.country).sum()
        e_missing = e_target - e_installed
        missing_mh_i = hydro.query("max_hours.isnull()").index

        if hydro_max_hours == "energy_capacity_totals_by_country":
            max_hours_country = (
                e_missing / hydro.loc[missing_mh_i].groupby("country").p_nom.sum()
            )

        elif hydro_max_hours == "estimate_by_large_installations":
            max_hours_country = (
                hydro_stats["E_store[TWh]"] * 1e3 / hydro_stats["p_nom_discharge[GW]"]
            )

        max_hours_country.clip(lower=0, inplace=True)

        missing_countries = pd.Index(hydro["country"].unique()).difference(
            max_hours_country.dropna().index
        )
        if not missing_countries.empty:
            logger.warning(
                "Assuming max_hours=6 for hydro reservoirs in the countries: {}".format(
                    ", ".join(missing_countries)
                )
            )
        hydro_max_hours_default = c.get("hydro_max_hours_default", 6.0)
        hydro_max_hours = hydro.max_hours.where(
            hydro.max_hours > 0, hydro.country.map(max_hours_country)
        ).fillna(hydro_max_hours_default)

        hydro_nominal = apply_powerplant_candidate_nominal_attrs(
            hydro, candidate_config, derating_config, "reservoir hydro"
        )
        hydro_candidate = hydro_nominal.pop("candidate")
        if hydro_candidate.any():
            logger.info(
                "Treating {} reservoir hydro units as build candidates with {:.2f} MW p_nom_max.".format(
                    hydro_candidate.sum(), hydro.loc[hydro_candidate, "p_nom"].sum()
                )
            )
        existing_hydro_capital_cost = (
            costs.at["hydro", "capital_cost"] if c.get("hydro_capital_cost") else 0.0
        )
        hydro_capital_cost = pd.Series(existing_hydro_capital_cost, index=hydro.index)
        hydro_capital_cost.loc[hydro_candidate] = costs.at["hydro", "capital_cost"]
        (
            hydro_annual_generation,
            hydro_full_load_hours,
            hydro_candidate_capacity,
        ) = get_hydro_project_energy_attrs(hydro, hydro_candidate, "reservoir hydro")
        hydro_inflow_t = inflow_t.loc[:, hydro.index].copy()
        if hydro_candidate.any():
            hydro_inflow_t = scale_hydro_candidate_inflow_timeseries(
                hydro_inflow_t,
                hydro,
                hydro_candidate,
                "reservoir hydro",
            )
            hydro_efficiency = costs.at["hydro", "efficiency"]
            hydro_inflow_t.loc[:, hydro_candidate] = (
                hydro_inflow_t.loc[:, hydro_candidate] / hydro_efficiency
            )

        n.madd(
            "StorageUnit",
            hydro.index,
            carrier="hydro",
            bus=hydro["bus"],
            **hydro_nominal,
            max_hours=hydro_max_hours,
            capital_cost=hydro_capital_cost,
            marginal_cost=costs.at["hydro", "marginal_cost"],
            p_max_pu=1.0,  # dispatch
            p_min_pu=0.0,  # store
            efficiency_dispatch=costs.at["hydro", "efficiency"],
            efficiency_store=0.0,
            cyclic_state_of_charge=True,
            inflow=hydro_inflow_t,
        )
        n.storage_units.loc[hydro.index, "annualgeneration_gwh"] = hydro_annual_generation
        n.storage_units.loc[hydro.index, "fullloadhours_h"] = hydro_full_load_hours
        n.storage_units.loc[hydro.index, "candidate_capacity_mw"] = (
            hydro_candidate_capacity
        )


def attach_extendable_generators(n, costs, ppl):
    logger.warning("The function is deprecated with the next release")
    elec_opts = snakemake.params.electricity
    carriers = pd.Index(elec_opts["extendable_carriers"]["Generator"])

    _add_missing_carriers_from_costs(n, costs, carriers)

    for tech in carriers:
        if tech.startswith("OCGT"):
            ocgt = (
                ppl.query("carrier in ['OCGT', 'CCGT']")
                .groupby("bus", as_index=False)
                .first()
            )
            n.madd(
                "Generator",
                ocgt.index,
                suffix=" OCGT",
                bus=ocgt["bus"],
                carrier=tech,
                p_nom_extendable=True,
                p_nom=0.0,
                capital_cost=costs.at["OCGT", "capital_cost"],
                marginal_cost=costs.at["OCGT", "marginal_cost"],
                efficiency=costs.at["OCGT", "efficiency"],
            )

        elif tech.startswith("CCGT"):
            ccgt = (
                ppl.query("carrier in ['OCGT', 'CCGT']")
                .groupby("bus", as_index=False)
                .first()
            )
            n.madd(
                "Generator",
                ccgt.index,
                suffix=" CCGT",
                bus=ccgt["bus"],
                carrier=tech,
                p_nom_extendable=True,
                p_nom=0.0,
                capital_cost=costs.at["CCGT", "capital_cost"],
                marginal_cost=costs.at["CCGT", "marginal_cost"],
                efficiency=costs.at["CCGT", "efficiency"],
            )

        elif tech.startswith("nuclear"):
            nuclear = (
                ppl.query("carrier == 'nuclear'").groupby("bus", as_index=False).first()
            )
            n.madd(
                "Generator",
                nuclear.index,
                suffix=" nuclear",
                bus=nuclear["bus"],
                carrier=tech,
                p_nom_extendable=True,
                p_nom=0.0,
                capital_cost=costs.at["nuclear", "capital_cost"],
                marginal_cost=costs.at["nuclear", "marginal_cost"],
                efficiency=costs.at["nuclear", "efficiency"],
            )

        else:
            raise NotImplementedError(
                f"Adding extendable generators for carrier "
                "'{tech}' is not implemented, yet. "
                "Only OCGT, CCGT and nuclear are allowed at the moment."
            )


def estimate_renewable_capacities_irena(
    n, estimate_renewable_capacities_config, countries_config
):
    stats = estimate_renewable_capacities_config["stats"]
    if not stats:
        return

    year = estimate_renewable_capacities_config["year"]
    tech_map = estimate_renewable_capacities_config["technology_mapping"]
    tech_keys = list(tech_map.keys())
    countries = countries_config

    p_nom_max = estimate_renewable_capacities_config["p_nom_max"]
    p_nom_min = estimate_renewable_capacities_config["p_nom_min"]

    if len(countries) == 0:
        return
    if len(tech_map) == 0:
        return

    if stats == "irena":
        capacities = pm.data.IRENASTAT().powerplant.convert_country_to_alpha2()
    else:
        logger.info(
            f"Selected renewable capacity estimation statistics {stats} is not available, applying greenfield scenario instead"
        )
        return

    # Check if countries are in country list of stats
    missing = list(set(countries).difference(capacities.Country.unique()))
    if missing:
        logger.info(
            f"The countries {missing} are not provided in the stats and hence not scaled"
        )

    capacities = capacities.query(
        "Year == @year and Technology in @tech_keys and Country in @countries"
    )
    capacities = capacities.groupby(["Technology", "Country"]).Capacity.sum()

    logger.info(
        f"Heuristics applied to distribute renewable capacities [MW] "
        f"{capacities.groupby('Country').sum()}"
    )

    for ppm_technology, techs in tech_map.items():
        if ppm_technology not in capacities.index:
            logger.info(
                f"technology {ppm_technology} is not provided by {stats} and therefore not estimated"
            )
            continue

        tech_capacities = capacities.loc[ppm_technology].reindex(
            countries, fill_value=0.0
        )
        tech_i = n.generators.query("carrier in @techs").index
        n.generators.loc[tech_i, "p_nom"] = (
            (
                n.generators_t.p_max_pu[tech_i].mean()
                * n.generators.loc[tech_i, "p_nom_max"]
            )
            # maximal yearly generation
            .groupby(n.generators.bus.map(n.buses.country))
            .transform(lambda s: normed(s) * tech_capacities.at[s.name])
            .where(lambda s: s > 0.1, 0.0)
        )  # only capacities above 100kW
        n.generators.loc[tech_i, "p_nom_min"] = n.generators.loc[tech_i, "p_nom"]

        if p_nom_min:
            assert np.isscalar(p_nom_min)
            logger.info(
                f"Scaling capacity stats to {p_nom_min*100:.2f}% of installed capacity acquired from stats."
            )
            n.generators.loc[tech_i, "p_nom_min"] = n.generators.loc[
                tech_i, "p_nom"
            ] * float(p_nom_min)

        if p_nom_max:
            assert np.isscalar(p_nom_max)
            logger.info(
                f"Scaling capacity expansion limit to {p_nom_max*100:.2f}% of installed capacity acquired from stats."
            )
            n.generators.loc[tech_i, "p_nom_max"] = n.generators.loc[
                tech_i, "p_nom_min"
            ] * float(p_nom_max)


def add_nice_carrier_names(n, config):
    carrier_i = n.carriers.index
    nice_names = (
        pd.Series(config["plotting"]["nice_names"])
        .reindex(carrier_i)
        .fillna(carrier_i.to_series().str.title())
    )
    n.carriers["nice_name"] = nice_names
    colors = pd.Series(config["plotting"]["tech_colors"]).reindex(carrier_i)
    if colors.isna().any():
        missing_i = list(colors.index[colors.isna()])
        logger.warning(
            f"tech_colors for carriers {missing_i} not defined " "in config."
        )
    n.carriers["color"] = colors


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("add_electricity")

    configure_logging(snakemake)

    n = pypsa.Network(snakemake.input.base_network)
    Nyears = n.snapshot_weightings.objective.sum() / 8760.0

    # Snakemake imports:
    demand_profiles = snakemake.input["demand_profiles"]

    costs = load_costs(
        snakemake.input.tech_costs,
        snakemake.params.costs,
        snakemake.params.electricity,
        Nyears,
    )
    ppl = load_powerplants(snakemake.input.powerplants)
    if "renewable_carriers" in snakemake.params.electricity:
        renewable_carriers = set(snakemake.params.electricity["renewable_carriers"])
    else:
        logger.warning(
            "Missing key `renewable_carriers` under config entry `electricity`. "
            "In future versions, this will raise an error. "
            "Falling back to carriers listed under `renewable`."
        )
        renewable_carriers = set(snakemake.params.renewable)

    extendable_carriers = snakemake.params.electricity["extendable_carriers"]
    candidate_config = get_powerplant_candidate_config(snakemake.params.electricity)
    derating_config = get_existing_capacity_derating_config(
        snakemake.params.electricity
    )
    if not (set(renewable_carriers) & set(extendable_carriers["Generator"])):
        logger.warning(
            "No renewables found in config entry `extendable_carriers`. "
            "In future versions, these have to be explicitly listed. "
            "Falling back to all renewables."
        )

    conventional_carriers = snakemake.params.electricity["conventional_carriers"]
    attach_load(n, demand_profiles)
    update_transmission_costs(n, costs, snakemake.params.length_factor)
    conventional_inputs = {
        k: v for k, v in snakemake.input.items() if k.startswith("conventional_")
    }
    attach_conventional_generators(
        n,
        costs,
        ppl,
        conventional_carriers,
        extendable_carriers,
        renewable_carriers,
        snakemake.params.conventional,
        snakemake.params.electricity.get("extendable_existing_carrier_limits", {}),
        conventional_inputs,
        candidate_config,
        derating_config,
    )
    attach_wind_and_solar(
        n,
        costs,
        ppl,
        snakemake.input,
        renewable_carriers,
        extendable_carriers,
        snakemake.params.length_factor,
    )
    attach_hydro(n, costs, ppl, candidate_config, derating_config)

    if snakemake.params.electricity.get("estimate_renewable_capacities"):
        estimate_renewable_capacities_irena(
            n,
            snakemake.params.electricity["estimate_renewable_capacities"],
            snakemake.params.countries,
        )

    update_p_nom_max(n)
    add_nice_carrier_names(n, snakemake.config)

    if not ("weight" in n.generators.columns):
        logger.warning(
            "Unexpected missing 'weight' column, which has been manually added. It may be due to missing generators."
        )
        n.generators["weight"] = pd.Series()

    sanitize_carriers(n, snakemake.config)
    if "location" in n.buses:
        sanitize_locations(n)

    n.meta = snakemake.config
    n.export_to_netcdf(snakemake.output[0])
