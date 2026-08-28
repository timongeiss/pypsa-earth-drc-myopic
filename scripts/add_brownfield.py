# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText:  PyPSA-Earth and PyPSA-Eur Authors
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Prepares brownfield data from previous planning horizon.
"""

import logging

import numpy as np
import pandas as pd
import pypsa
import xarray as xr
from _helpers import sanitize_carriers, sanitize_locations, read_csv_nafix
from add_existing_baseyear import add_build_year_to_new_assets

# from pypsa.clustering.spatial import normed_or_uniform

logger = logging.getLogger(__name__)
idx = pd.IndexSlice

HYDRO_BROWNFIELD_MARKER = "brownfield_hydro_candidate"
HYDRO_BROWNFIELD_COMPONENTS = {
    "Generator": {
        "list_name": "generators",
        "attr": "p",
        "carriers": ["ror"],
    },
    "StorageUnit": {
        "list_name": "storage_units",
        "attr": "p",
        "carriers": ["hydro", "PHS"],
    },
}

COMPONENT_LIST_NAMES = {
    "Bus": "buses",
    "Generator": "generators",
    "StorageUnit": "storage_units",
    "Link": "links",
    "Store": "stores",
}


def _series_input_attrs(n, component_name):
    selection = n.component_attrs[component_name].type.str.contains(
        "series"
    ) & n.component_attrs[component_name].status.str.contains("Input")
    return n.component_attrs[component_name].index[selection]


def _bool_series(component_df, column):
    if column not in component_df:
        return pd.Series(False, index=component_df.index)
    return component_df[column].fillna(False).astype(bool)


def _hydro_brownfield_nominal(component_df, attr):
    opt = pd.to_numeric(
        component_df.get(
            f"{attr}_nom_opt", pd.Series(np.nan, index=component_df.index)
        ),
        errors="coerce",
    ).fillna(0.0)
    fixed = pd.to_numeric(
        component_df.get(f"{attr}_nom", pd.Series(0.0, index=component_df.index)),
        errors="coerce",
    ).fillna(0.0)
    marked = _bool_series(component_df, HYDRO_BROWNFIELD_MARKER)

    # Extendable candidates are carried over by their optimized capacity only.
    # Previously imported fixed hydro candidates use p_nom if p_nom_opt is absent.
    return opt.where(~marked, opt.where(opt > 0.0, fixed))


def _branch_component_removals(config, h2export):
    removals = (
        config.get("export", {})
        .get("branch_component_removals", {})
        or {}
    )
    label_keys = [h2export, str(h2export)]
    try:
        label_keys.append(float(h2export))
    except (TypeError, ValueError):
        pass

    for key in label_keys:
        if key in removals:
            return removals[key] or {}

    return {}


def _remove_branch_components(n, config, h2export):
    for component_name, names in _branch_component_removals(config, h2export).items():
        list_name = COMPONENT_LIST_NAMES.get(component_name)
        if list_name is None:
            raise ValueError(
                f"Unsupported component '{component_name}' in "
                "export.branch_component_removals."
            )

        if isinstance(names, str):
            names = [names]

        requested = pd.Index(names)
        component_df = getattr(n, list_name)
        existing = requested.intersection(component_df.index)
        missing = requested.difference(component_df.index)

        if not missing.empty:
            logger.warning(
                "Configured branch component removals for h2export=%s include "
                "missing %s assets: %s.",
                h2export,
                component_name,
                ", ".join(map(str, missing)),
            )

        if existing.empty:
            continue

        n.mremove(component_name, existing)
        logger.info(
            "Removed %d %s assets for h2export=%s: %s.",
            len(existing),
            component_name,
            h2export,
            ", ".join(map(str, existing)),
        )


def _carrier_phaseout_removals(config, year):
    phaseouts = (
        config.get("existing_capacities", {})
        .get("brownfield_carrier_phaseout", {})
        or {}
    )

    for key in (year, str(year)):
        if key in phaseouts:
            return phaseouts[key] or {}

    return {}


def _unused_buses_with_carriers(n, carriers):
    if n.buses.empty or "carrier" not in n.buses.columns:
        return pd.Index([])

    carrier_index = pd.Index([str(carrier) for carrier in carriers])
    candidates = n.buses.index[
        n.buses.carrier.fillna("").astype(str).isin(carrier_index)
    ]
    if candidates.empty:
        return candidates

    references = []
    for list_name in ("loads", "generators", "storage_units", "stores"):
        component_df = getattr(n, list_name)
        if "bus" in component_df.columns:
            references.extend(component_df["bus"].dropna().astype(str).tolist())

    for list_name in ("links", "lines", "transformers"):
        component_df = getattr(n, list_name)
        for column in component_df.columns:
            if column.startswith("bus"):
                references.extend(component_df[column].dropna().astype(str).tolist())

    referenced = pd.Index(references)
    return candidates[~candidates.astype(str).isin(referenced)]


def _remove_carrier_components(n, carrier_removals, context):
    # Remove buses last so carrier assets using those buses can be removed first.
    items = sorted(carrier_removals.items(), key=lambda item: item[0] == "Bus")

    for component_name, carriers in items:
        list_name = COMPONENT_LIST_NAMES.get(component_name)
        if list_name is None:
            raise ValueError(
                f"Unsupported component '{component_name}' in "
                "existing_capacities.brownfield_carrier_phaseout."
            )

        if isinstance(carriers, str):
            carriers = [carriers]

        component_df = getattr(n, list_name)
        if component_df.empty or "carrier" not in component_df.columns:
            continue

        if component_name == "Bus":
            to_remove = _unused_buses_with_carriers(n, carriers)
        else:
            carrier_index = pd.Index([str(carrier) for carrier in carriers])
            to_remove = component_df.index[
                component_df.carrier.fillna("").astype(str).isin(carrier_index)
            ]

        if to_remove.empty:
            continue

        n.mremove(component_name, to_remove)
        logger.info(
            "Removed %d %s assets with carrier(s) %s from %s.",
            len(to_remove),
            component_name,
            ", ".join(map(str, carriers)),
            context,
        )


def _carry_over_hydro_candidate_capacity(n, n_p, threshold):
    """
    Carry optimized hydro project candidates into the next myopic horizon.

    Hydro project candidates are re-created from the horizon-specific powerplant
    table. For matching project names, keep the current-horizon candidate and set
    the previous optimized capacity as minimum capacity. For projects no longer
    present in the current horizon, import the previous asset as fixed capacity.
    """
    threshold = float(threshold)

    for component_name, spec in HYDRO_BROWNFIELD_COMPONENTS.items():
        current_df = getattr(n, spec["list_name"])
        previous_df = getattr(n_p, spec["list_name"])
        attr = spec["attr"]

        if previous_df.empty:
            continue

        previous_nominal = _hydro_brownfield_nominal(previous_df, attr)
        previous_extendable = _bool_series(previous_df, f"{attr}_nom_extendable")
        previous_marked = _bool_series(previous_df, HYDRO_BROWNFIELD_MARKER)
        previous_candidate = (
            previous_df.carrier.isin(spec["carriers"])
            & (previous_extendable | previous_marked)
            & (previous_nominal >= threshold)
        )
        previous_i = previous_df.index[previous_candidate]

        if previous_i.empty:
            continue

        same_i = previous_i.intersection(current_df.index)
        if not same_i.empty:
            min_col = f"{attr}_nom_min"
            max_col = f"{attr}_nom_max"
            nom_col = f"{attr}_nom"

            previous_min = previous_nominal.loc[same_i]
            current_min = pd.to_numeric(
                current_df.loc[same_i, min_col], errors="coerce"
            ).fillna(0.0)
            current_nom = pd.to_numeric(
                current_df.loc[same_i, nom_col], errors="coerce"
            ).fillna(0.0)

            new_min = pd.concat([current_min, previous_min], axis=1).max(axis=1)
            new_nom = pd.concat([current_nom, new_min], axis=1).max(axis=1)

            current_df.loc[same_i, min_col] = new_min
            current_df.loc[same_i, nom_col] = new_nom
            current_df.loc[same_i, HYDRO_BROWNFIELD_MARKER] = True

            if max_col in current_df:
                current_max = pd.to_numeric(
                    current_df.loc[same_i, max_col], errors="coerce"
                )
                finite_max = pd.Series(
                    np.isfinite(current_max.to_numpy(dtype=float)),
                    index=current_max.index,
                )
                below_min = finite_max & (current_max < new_min)
                if below_min.any():
                    current_df.loc[below_min.index[below_min], max_col] = new_min.loc[
                        below_min
                    ]
                    logger.warning(
                        "Raised %s for %d %s hydro candidates to their carried-over "
                        "minimum capacity because the current project maximum was lower.",
                        max_col,
                        int(below_min.sum()),
                        component_name,
                    )

            logger.info(
                "Set carried-over minimum capacity for %d %s hydro candidates "
                "(%.2f MW).",
                len(same_i),
                component_name,
                previous_min.sum(),
            )

        missing_i = previous_i.difference(current_df.index)
        if not missing_i.empty:
            fixed = previous_df.loc[missing_i].copy()
            fixed_nominal = previous_nominal.loc[missing_i]
            fixed[f"{attr}_nom"] = fixed_nominal
            fixed[f"{attr}_nom_min"] = fixed_nominal
            if f"{attr}_nom_max" in fixed:
                fixed[f"{attr}_nom_max"] = fixed_nominal
            fixed[f"{attr}_nom_extendable"] = False
            fixed[HYDRO_BROWNFIELD_MARKER] = True

            n.import_components_from_dataframe(fixed, component_name)

            previous_pnl = getattr(n_p, spec["list_name"] + "_t")
            for tattr in _series_input_attrs(n, component_name):
                if not hasattr(previous_pnl, tattr):
                    continue
                data = getattr(previous_pnl, tattr)
                cols = missing_i.intersection(data.columns)
                if cols.empty:
                    continue
                n.import_series_from_dataframe(data.loc[:, cols], component_name, tattr)

            logger.info(
                "Imported %d missing previous %s hydro candidates as fixed "
                "brownfield capacity (%.2f MW).",
                len(missing_i),
                component_name,
                fixed_nominal.sum(),
            )

        # Remove handled assets from n_p so the generic brownfield import below
        # neither drops nor duplicates them.
        n_p.mremove(component_name, previous_i)


def add_brownfield(n, n_p, year):
    logger.info(f"Preparing brownfield for the year {year}")

    # electric transmission grid set optimised capacities of previous as minimum
    n.lines.s_nom_min = n_p.lines.s_nom_opt
    dc_i = n.links[n.links.carrier == "DC"].index
    n.links.loc[dc_i, "p_nom_min"] = n_p.links.loc[dc_i, "p_nom_opt"]

    _remove_branch_components(n, snakemake.config, snakemake.wildcards.h2export)

    carrier_phaseout = _carrier_phaseout_removals(snakemake.config, year)
    if carrier_phaseout:
        _remove_carrier_components(n, carrier_phaseout, f"current {year} network")
        _remove_carrier_components(
            n_p, carrier_phaseout, f"previous network imported into {year}"
        )

    threshold = snakemake.params.threshold_capacity
    _carry_over_hydro_candidate_capacity(n, n_p, threshold)

    for c in n_p.iterate_components(["Link", "Generator", "Store"]):
        attr = "e" if c.name == "Store" else "p"

        # first, remove generators, links and stores that track
        # CO2 or global EU values since these are already in n
        n_p.mremove(c.name, c.df.index[c.df.lifetime == np.inf])

        # remove assets whose build_year + lifetime < year
        n_p.mremove(c.name, c.df.index[c.df.build_year + c.df.lifetime < year])

        # remove assets if their optimized nominal capacity is lower than a threshold
        # since CHP heat Link is proportional to CHP electric Link, make sure threshold is compatible
        chp_heat = c.df.index[
            (c.df[f"{attr}_nom_extendable"] & c.df.index.str.contains("urban central"))
            & c.df.index.str.contains("CHP")
            & c.df.index.str.contains("heat")
        ]

        if not chp_heat.empty:
            threshold_chp_heat = (
                threshold
                * c.df.efficiency[chp_heat.str.replace("heat", "electric")].values
                * c.df.p_nom_ratio[chp_heat.str.replace("heat", "electric")].values
                / c.df.efficiency[chp_heat].values
            )
            n_p.mremove(
                c.name,
                chp_heat[c.df.loc[chp_heat, f"{attr}_nom_opt"] < threshold_chp_heat],
            )

        n_p.mremove(
            c.name,
            c.df.index[
                (c.df[f"{attr}_nom_extendable"] & ~c.df.index.isin(chp_heat))
                & (c.df[f"{attr}_nom_opt"] < threshold)
            ],
        )

        # copy over assets but fix their capacity
        c.df[f"{attr}_nom"] = c.df[f"{attr}_nom_opt"]
        c.df[f"{attr}_nom_extendable"] = False

        # remove assets if name already exist in the new network
        n_p.mremove(c.name, c.df.index.intersection(getattr(n, c.list_name).index))

        n.import_components_from_dataframe(c.df, c.name)

        # copy time-dependent
        selection = n.component_attrs[c.name].type.str.contains(
            "series"
        ) & n.component_attrs[c.name].status.str.contains("Input")
        for tattr in n.component_attrs[c.name].index[selection]:
            n.import_series_from_dataframe(c.pnl[tattr], c.name, tattr)

        # deal with gas network
        pipe_carrier = ["gas pipeline"]
        if snakemake.params.H2_retrofit:
            # drop capacities of previous year to avoid duplicating
            to_drop = n.links.carrier.isin(pipe_carrier) & (n.links.build_year != year)
            n.mremove("Link", n.links.loc[to_drop].index)

            # subtract the already retrofitted from today's gas grid capacity
            h2_retrofitted_fixed_i = n.links[
                (n.links.carrier == "H2 pipeline retrofitted")
                & (n.links.build_year != year)
            ].index
            gas_pipes_i = n.links[n.links.carrier.isin(pipe_carrier)].index
            CH4_per_H2 = 1 / snakemake.params.H2_retrofit_capacity_per_CH4
            fr = "H2 pipeline retrofitted"
            to = "gas pipeline"
            # today's pipe capacity
            pipe_capacity = n.links.loc[gas_pipes_i, "p_nom"]
            # already retrofitted capacity from gas -> H2
            already_retrofitted = (
                n.links.loc[h2_retrofitted_fixed_i, "p_nom"]
                .rename(lambda x: x.split("-2")[0].replace(fr, to))
                .groupby(level=0)
                .sum()
            )
            remaining_capacity = (
                pipe_capacity
                - CH4_per_H2
                * already_retrofitted.reindex(index=pipe_capacity.index).fillna(0)
            )
            n.links.loc[gas_pipes_i, "p_nom"] = remaining_capacity
        else:
            new_pipes = n.links.carrier.isin(pipe_carrier) & (
                n.links.build_year == year
            )
            n.links.loc[new_pipes, "p_nom"] = 0.0
            n.links.loc[new_pipes, "p_nom_min"] = 0.0


def disable_grid_expansion_if_limit_hit(n):
    """
    Check if transmission expansion limit is already reached; then turn off.

    In particular, this function checks if the total transmission
    capital cost or volume implied by s_nom_min and p_nom_min are
    numerically close to the respective global limit set in
    n.global_constraints. If so, the nominal capacities are set to the
    minimum and extendable is turned off; the corresponding global
    constraint is then dropped.
    """
    cols = {"cost": "capital_cost", "volume": "length"}
    for limit_type in ["cost", "volume"]:
        glcs = n.global_constraints.query(
            f"type == 'transmission_expansion_{limit_type}_limit'"
        )

        for name, glc in glcs.iterrows():
            total_expansion = (
                (
                    n.lines.query("s_nom_extendable")
                    .eval(f"s_nom_min * {cols[limit_type]}")
                    .sum()
                )
                + (
                    n.links.query("carrier == 'DC' and p_nom_extendable")
                    .eval(f"p_nom_min * {cols[limit_type]}")
                    .sum()
                )
            ).sum()

            # Allow small numerical differences
            if np.abs(glc.constant - total_expansion) / glc.constant < 1e-6:
                logger.info(
                    f"Transmission expansion {limit_type} is already reached, disabling expansion and limit"
                )
                extendable_acs = n.lines.query("s_nom_extendable").index
                n.lines.loc[extendable_acs, "s_nom_extendable"] = False
                n.lines.loc[extendable_acs, "s_nom"] = n.lines.loc[
                    extendable_acs, "s_nom_min"
                ]

                extendable_dcs = n.links.query(
                    "carrier == 'DC' and p_nom_extendable"
                ).index
                n.links.loc[extendable_dcs, "p_nom_extendable"] = False
                n.links.loc[extendable_dcs, "p_nom"] = n.links.loc[
                    extendable_dcs, "p_nom_min"
                ]

                n.global_constraints.drop(name, inplace=True)


# def adjust_renewable_profiles(n, input_profiles, params, year):
#     """
#     Adjusts renewable profiles according to the renewable technology specified,
#     using the latest year below or equal to the selected year.
#     """

#     # spatial clustering
#     cluster_busmap = read_csv_nafix(snakemake.input.cluster_busmap, index_col=0).squeeze()
#     simplify_busmap = read_csv_nafix(
#         snakemake.input.simplify_busmap, index_col=0
#     ).squeeze()
#     clustermaps = simplify_busmap.map(cluster_busmap)
#     clustermaps.index = clustermaps.index.astype(str)

#     # temporal clustering
#     dr = pd.date_range(**params["snapshots"], freq="h")
#     snapshotmaps = (
#         pd.Series(dr, index=dr).where(lambda x: x.isin(n.snapshots), pd.NA).ffill()
#     )

#     for carrier in params["carriers"]:
#         if carrier == "hydro":
#             continue
#         with xr.open_dataset(getattr(input_profiles, "profile_" + carrier)) as ds:
#             if ds.indexes["bus"].empty or "year" not in ds.indexes:
#                 continue

#             closest_year = max(
#                 (y for y in ds.year.values if y <= year), default=min(ds.year.values)
#             )

#             p_max_pu = (
#                 ds["profile"]
#                 .sel(year=closest_year)
#                 .transpose("time", "bus")
#                 .to_pandas()
#             )

#             # spatial clustering
#             weight = ds["weight"].sel(year=closest_year).to_pandas()
#             weight = weight.groupby(clustermaps).transform(normed_or_uniform)
#             p_max_pu = (p_max_pu * weight).T.groupby(clustermaps).sum().T
#             p_max_pu.columns = p_max_pu.columns + f" {carrier}"

#             # temporal_clustering
#             p_max_pu = p_max_pu.groupby(snapshotmaps).mean()

#             # replace renewable time series
#             n.generators_t.p_max_pu.loc[:, p_max_pu.columns] = p_max_pu


if __name__ == "__main__":
    if "snakemake" not in globals():

        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "add_brownfield",
            simpl="",
            clusters="4",
            ll="c1",
            opts="Co2L-4H",
            planning_horizons="2030",
            sopts="144H",
            discountrate=0.071,
            demand="AB",
            h2export="120",
        )

    logger.info(f"Preparing brownfield from the file {snakemake.input.network_p}")

    year = int(snakemake.wildcards.planning_horizons)

    n = pypsa.Network(snakemake.input.network)

    # TODO
    # adjust_renewable_profiles(n, snakemake.input, snakemake.params, year)

    add_build_year_to_new_assets(n, year)

    n_p = pypsa.Network(snakemake.input.network_p)

    add_brownfield(n, n_p, year)

    disable_grid_expansion_if_limit_hit(n)

    sanitize_carriers(n, snakemake.config)
    sanitize_locations(n)

    n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
    n.export_to_netcdf(snakemake.output[0])
