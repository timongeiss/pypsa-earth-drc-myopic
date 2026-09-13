# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText:  PyPSA-Earth and PyPSA-Eur Authors
# SPDX-FileCopyrightText:  2026 Timon Geiss
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# -*- coding: utf-8 -*-
"""
Proposed code structure:
X read network (.nc-file)
X add export bus
X connect hydrogen buses (advanced: only ports, not all) to export bus
X add store and connect to export bus
X (add load and connect to export bus) only required if the "store" option fails

Possible improvements:
- Select port buses automatically (with both voronoi and gadm clustering). Use data/ports.csv?
"""


import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import pypsa
from _helpers import (
    locate_bus,
    override_component_attrs,
    prepare_costs,
    read_csv_nafix,
    resolve_h2export_for_planning_horizon,
    resolve_snakemake_config_by_planning_horizon,
)

logger = logging.getLogger(__name__)


COMPONENT_LIST_NAMES = {
    "Bus": "buses",
    "Generator": "generators",
    "Link": "links",
    "Load": "loads",
    "StorageUnit": "storage_units",
    "Store": "stores",
}

COMPONENT_BUS_COLUMNS = {
    "Bus": [],
    "Generator": ["bus"],
    "Link": ["bus0", "bus1", "bus2", "bus3", "bus4"],
    "Load": ["bus"],
    "StorageUnit": ["bus"],
    "Store": ["bus"],
}


def optional_single_input(value):
    if value is None:
        return None
    if isinstance(value, str):
        return value or None
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else None
    return str(value)


def series_input_attrs(n, component_name):
    attrs = n.component_attrs[component_name]
    selection = attrs.type.str.contains("series", na=False) & attrs.status.str.contains(
        "Input", na=False
    )
    return attrs.index[selection]


def validate_component_buses(n, component_name, component_df):
    missing = []
    for column in COMPONENT_BUS_COLUMNS.get(component_name, []):
        if column not in component_df:
            continue
        buses = component_df[column].dropna().astype(str)
        buses = buses[buses != ""]
        missing.extend(buses[~buses.isin(n.buses.index)].unique())

    if missing:
        raise ValueError(
            f"Branch component additions for {component_name} reference missing "
            f"target buses: {', '.join(map(str, sorted(set(missing))))}"
        )


def normalise_component_names(names):
    if isinstance(names, str):
        return [names]
    return list(names or [])


def apply_hydro_candidate_costs(component_name, component_df, costs):
    if component_name != "StorageUnit" or component_df.empty:
        return component_df
    if "carrier" not in component_df:
        return component_df

    hydro = component_df["carrier"].astype(str).eq("hydro")
    if not hydro.any():
        return component_df

    if "hydro" not in costs.index or "fixed" not in costs.columns:
        raise KeyError(
            "Cannot set early_large hydro candidate costs; costs['hydro', 'fixed'] "
            "is missing."
        )

    component_df = component_df.copy()
    component_df.loc[hydro, "capital_cost"] = costs.at["hydro", "fixed"]
    component_df.loc[hydro, "p_nom"] = 0.0
    component_df.loc[hydro, "p_nom_min"] = 0.0
    component_df.loc[hydro, "p_nom_extendable"] = True
    if "p_nom_opt" in component_df:
        component_df.loc[hydro, "p_nom_opt"] = 0.0
    return component_df


def add_branch_component_additions(n, costs, overrides):
    additions = getattr(snakemake.params, "branch_component_additions", {}) or {}
    components_to_add = additions.get("components", {}) or {}
    if not components_to_add:
        return False

    template_path = optional_single_input(
        snakemake.input.get("branch_template_network")
    )
    if not template_path:
        raise ValueError(
            "Branch component additions are configured, but no template network "
            "input was provided."
        )

    template = pypsa.Network(template_path, override_component_attrs=overrides)
    if not template.snapshots.equals(n.snapshots):
        raise ValueError(
            "Branch component addition template snapshots do not match target "
            f"network snapshots: {template_path}"
        )

    for component_name, names in components_to_add.items():
        list_name = COMPONENT_LIST_NAMES.get(component_name)
        if list_name is None:
            raise ValueError(
                f"Unsupported component '{component_name}' in "
                "export.branch_component_additions."
            )

        requested = pd.Index(normalise_component_names(names))
        if requested.empty:
            continue

        template_df = getattr(template, list_name)
        target_df = getattr(n, list_name)
        missing = requested.difference(template_df.index)
        existing = requested.intersection(target_df.index)

        if not missing.empty:
            raise ValueError(
                f"Branch component addition template {template_path} is missing "
                f"{component_name} assets: {', '.join(map(str, missing))}"
            )
        if not existing.empty:
            raise ValueError(
                f"Branch component additions would overwrite existing {component_name} "
                f"assets: {', '.join(map(str, existing))}"
            )

        component_df = template_df.loc[requested].copy()
        component_df = apply_hydro_candidate_costs(component_name, component_df, costs)
        validate_component_buses(n, component_name, component_df)
        n.import_components_from_dataframe(component_df, component_name)

        template_pnl = getattr(template, list_name + "_t")
        for attr in series_input_attrs(n, component_name):
            if not hasattr(template_pnl, attr):
                continue
            series = getattr(template_pnl, attr)
            cols = requested.intersection(series.columns)
            if cols.empty:
                continue
            n.import_series_from_dataframe(series.loc[:, cols], component_name, attr)

        logger.info(
            "Added %d %s branch components from %s: %s.",
            len(requested),
            component_name,
            template_path,
            ", ".join(map(str, requested)),
        )

    return True


def select_ports(n):
    """
    This function selects the buses where ports are located.
    """

    ports = read_csv_nafix(
        snakemake.input.export_ports,
        index_col=None,
        keep_default_na=False,
    )

    if "bus" in ports.columns:
        bus_ids = ports["bus"].astype(str).str.strip()
        bus_ids = pd.Index(bus_ids[bus_ids != ""]).drop_duplicates()

        if bus_ids.empty:
            raise ValueError(
                "The export ports file contains a 'bus' column, but no bus IDs were provided."
            )

        h2_bus_ids = pd.Index(
            [bus if bus.endswith(" H2") else f"{bus} H2" for bus in bus_ids]
        )
        missing = h2_bus_ids.difference(n.buses.index)
        if not missing.empty:
            raise ValueError(
                "The following explicit hydrogen export buses are missing from the "
                f"network: {', '.join(missing)}"
            )

        non_h2 = n.buses.loc[h2_bus_ids].index[
            n.buses.loc[h2_bus_ids, "carrier"] != "H2"
        ]
        if len(non_h2) > 0:
            raise ValueError(
                "The following explicit export buses exist but are not H2 buses: "
                f"{', '.join(non_h2)}"
            )

        hydrogen_buses_ports = n.buses.loc[h2_bus_ids]
        logger.info(
            "Using explicit hydrogen export buses from export_ports.csv: %s",
            ", ".join(hydrogen_buses_ports.index),
        )
        hydrogen_buses_ports.index.name = "Bus"
        return hydrogen_buses_ports

    gadm_layer_id = snakemake.params.gadm_layer_id

    ports = locate_bus(
        ports,
        countries,
        gadm_layer_id,
        snakemake.input.shapes_path,
        snakemake.params.alternative_clustering,
    )

    # TODO: revise if ports quantity and property by shape become relevant
    # drop duplicated entries
    gcol = "gadm_{}".format(gadm_layer_id)
    ports_sel = ports.loc[~ports[gcol].duplicated(keep="first")].set_index(gcol)

    # Select the hydrogen buses based on nodes with ports. If no ports exist, print info and set all nodes as export
    if ports_sel.empty:
        hydrogen_buses_ports = n.buses[n.buses.carrier == "H2"]
        logger.info(
            "No hydrogen export ports are found. Setting all hydrogen buses as export nodes"
        )
    else:
        hydrogen_buses_ports = n.buses.loc[ports_sel.index + " H2"]

    hydrogen_buses_ports.index.name = "Bus"

    return hydrogen_buses_ports


def add_export(n, hydrogen_buses_ports, export_profile):
    country_shape = gpd.read_file(snakemake.input["shapes_path"])
    # Find most northwestern point in country shape and get x and y coordinates
    country_shape = country_shape.to_crs(
        "EPSG:3395"
    )  # Project to Mercator projection (Projected)

    # Get coordinates of the most western and northern point of the country and add a buffer of 2 degrees (equiv. to approx 220 km)
    x_export = country_shape.geometry.centroid.x.min() - 2
    y_export = country_shape.geometry.centroid.y.max() + 2

    # add export bus
    n.add(
        "Bus",
        "H2 export bus",
        carrier="H2",
        location="Earth",
        x=x_export,
        y=y_export,
    )

    # add export links
    logger.info("Adding export links")
    n.madd(
        "Link",
        names=hydrogen_buses_ports.index + " export",
        bus0=hydrogen_buses_ports.index,
        bus1="H2 export bus",
        p_nom_extendable=True,
    )

    export_links = n.links[n.links.index.str.contains("export")]
    logger.info(export_links)

    # add store depending on config settings

    if snakemake.params.store == True:
        if snakemake.params.store_capital_costs == "no_costs":
            capital_cost = 0
        elif snakemake.params.store_capital_costs == "standard_costs":
            capital_cost = costs.at[
                "hydrogen storage tank type 1 including compressor", "fixed"
            ]
        else:
            logger.error(
                f"Value {snakemake.params.store_capital_costs} for ['export']['store_capital_costs'] is not valid"
            )

        n.add(
            "Store",
            "H2 export store",
            bus="H2 export bus",
            e_nom_extendable=True,
            carrier="H2",
            e_initial=0,  # actually not required, since e_cyclic=True
            marginal_cost=0,
            capital_cost=capital_cost,
            e_cyclic=True,
        )

    elif snakemake.params.store == False:
        pass

    if snakemake.params.export_endogenous:
        # add endogenous export by implementing a negative generation
        n.add(
            "Generator",
            "H2 export load",
            bus="H2 export bus",
            carrier="H2 export",
            sign=-1,
            p_nom_extendable=True,
            marginal_cost=snakemake.params.endogenous_price * (-1),
        )

    else:
        # add exogenous export by implementing a load
        n.add(
            "Load",
            "H2 export load",
            bus="H2 export bus",
            carrier="H2 export",
            p_set=export_profile,
        )

    return


def create_export_profile():
    """
    This function creates the export profile based on the annual export demand
    and resamples it to temp resolution obtained from the wildcard.
    """

    export_h2_twh = resolve_h2export_for_planning_horizon(
        snakemake.config,
        snakemake.wildcards["h2export"],
        snakemake.wildcards["planning_horizons"],
    )

    # convert TWh to MWh
    export_h2 = export_h2_twh * 1e6

    if snakemake.params.export_profile == "constant":
        export_profile = export_h2 / 8760
        snapshots = pd.date_range(freq="h", **snakemake.params.snapshots)
        export_profile = pd.Series(export_profile, index=snapshots)

    elif snakemake.params.export_profile == "ship":
        # Import hydrogen export ship profile and check if it matches the export demand obtained from the wildcard
        export_profile = read_csv_nafix(snakemake.input.ship_profile, index_col=0)
        export_profile.index = pd.to_datetime(export_profile.index)
        export_profile = pd.Series(
            export_profile["profile"], index=pd.to_datetime(export_profile.index)
        )

        if np.abs(export_profile.sum() - export_h2) > 1:  # Threshold of 1 MWh
            logger.error(
                f"Sum of ship profile ({export_profile.sum()/1e6} TWh) does not match export demand ({export_h2_twh} TWh)"
            )
            raise ValueError(
                f"Sum of ship profile ({export_profile.sum()/1e6} TWh) does not match export demand ({export_h2_twh} TWh)"
            )

    # Resample to temporal resolution defined in wildcard "sopts" with pandas resample
    sopts = snakemake.wildcards.sopts.split("-")
    export_profile = export_profile.resample(sopts[0].casefold()).mean()

    # revise logger msg
    export_type = snakemake.params.export_profile
    logger.info(
        f"The yearly export demand is {export_h2_twh} TWh, profile generated based on {export_type} method and resampled to {sopts[0]}"
    )

    return export_profile


if __name__ == "__main__":
    if "snakemake" not in globals():

        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "add_export",
            simpl="",
            clusters="10",
            ll="copt",
            opts="Co2L-144H",
            planning_horizons="2030",
            sopts="3H",
            discountrate="0.071",
            demand="AP",
            h2export="3",
            # configfile="test/config.test1.yaml",
        )

    resolve_snakemake_config_by_planning_horizon(snakemake)

    overrides = override_component_attrs(snakemake.input.overrides)
    n = pypsa.Network(snakemake.input.network, override_component_attrs=overrides)
    countries = list(n.buses.country[n.buses.country != ""].unique())

    export_h2_twh = resolve_h2export_for_planning_horizon(
        snakemake.config,
        snakemake.wildcards["h2export"],
        snakemake.wildcards["planning_horizons"],
    )
    branch_additions = getattr(snakemake.params, "branch_component_additions", {}) or {}
    has_branch_additions = bool(branch_additions.get("components", {}))

    if export_h2_twh == 0 and not has_branch_additions:
        logger.info("Resolved H2 export is 0 TWh. Skipping export components.")
        n.export_to_netcdf(snakemake.output[0])
        logger.info("Network successfully exported")
        raise SystemExit(0)

    # Prepare the costs dataframe
    Nyears = n.snapshot_weightings.generators.sum() / 8760

    costs = prepare_costs(
        snakemake.input.costs,
        snakemake.config["costs"],
        snakemake.params.costs["output_currency"],
        snakemake.params.costs["fill_values"],
        Nyears,
        snakemake.params.costs["default_exchange_rate"],
        snakemake.params.costs["future_exchange_rate_strategy"],
        snakemake.params.costs["custom_future_exchange_rate"],
    )

    branch_components_added = add_branch_component_additions(n, costs, overrides)

    if export_h2_twh == 0:
        if branch_components_added:
            logger.info(
                "Resolved H2 export is 0 TWh. Exporting network after branch "
                "component additions."
            )
        else:
            logger.info("Resolved H2 export is 0 TWh. Skipping export components.")
        n.export_to_netcdf(snakemake.output[0])
        logger.info("Network successfully exported")
        raise SystemExit(0)

    # Create export profile
    export_profile = create_export_profile()

    # get hydrogen export buses/ports
    hydrogen_buses_ports = select_ports(n)

    # add export value and components to network
    add_export(n, hydrogen_buses_ports, export_profile)

    n.export_to_netcdf(snakemake.output[0])

    logger.info("Network successfully exported")
