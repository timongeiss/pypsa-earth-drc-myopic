# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText:  PyPSA-Earth and PyPSA-Eur Authors
# SPDX-FileCopyrightText:  2026 Timon Geiss
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# -*- coding: utf-8 -*-
"""
Creates electric demand profile csv.

Relevant Settings
-----------------

.. code:: yaml

    load:
        scale:
        ssp:
        weather_year:
        prediction_year:
        region_load:

Inputs
------

- ``networks/base.nc``: confer :ref:`base`, a base PyPSA Network
- ``resources/bus_regions/regions_onshore.geojson``: confer :mod:`build_bus_regions`
- ``load_data_paths``: paths to load profiles, e.g. hourly country load profiles produced by GEGIS
- ``resources/shapes/gadm_shapes.geojson``: confer :ref:`shapes`, file containing the gadm shapes

Outputs
-------

- ``resources/demand_profiles.csv``: the content of the file is the electric demand profile associated to each bus. The file has the snapshots as rows and the buses of the network as columns.

Description
-----------

The rule :mod:`build_demand` creates load demand profiles in correspondence of the buses of the network.
It creates the load paths for GEGIS outputs by combining the input parameters of the countries, weather year, prediction year, and SSP scenario.
Then with a function that takes in the PyPSA network "base.nc", region and gadm shape data, the countries of interest, a scale factor, and the snapshots,
it returns a csv file called "demand_profiles.csv", that allocates the load to the buses of the network according to GDP and population.
"""
import os
import os.path
from itertools import product

import geopandas as gpd
import numpy as np
import pandas as pd
import pypsa
import scipy.sparse as sparse
import xarray as xr
from _helpers import (
    BASE_DIR,
    configure_logging,
    create_logger,
    read_csv_nafix,
    read_osm_config,
)
from shapely.geometry import Polygon
from shapely.prepared import prep
from shapely.validation import make_valid

logger = create_logger(__name__)


def normed(s):
    if s.sum() == 0:
        return s
    return s / s.sum()


def get_gegis_regions(countries):
    """
    Get the GEGIS region from the config file.

    Parameters
    ----------
    region : str
        The region of the bus

    Returns
    -------
    str
        The GEGIS region
    """
    gegis_dict, world_iso = read_osm_config("gegis_regions", "world_iso")

    regions = []

    for d_region in [gegis_dict, world_iso]:
        for key, value in d_region.items():
            # ignore if the key is already in the regions list
            if key not in regions:
                # if a country is in the regions values, then load it
                cintersect = set(countries).intersection(set(value.keys()))
                if cintersect:
                    regions.append(key)
    return regions


def get_load_paths_gegis(ssp_parentfolder, config):
    """
    Create load paths for GEGIS outputs.

    The paths are created automatically according to included country,
    weather year, prediction year and ssp scenario

    Example
    -------
    ["/data/ssp2-2.6/2030/era5_2013/Africa.nc", "/data/ssp2-2.6/2030/era5_2013/Africa.nc"]
    """
    countries = config.get("countries")
    region_load = get_gegis_regions(countries)
    weather_year = config.get("load_options")["weather_year"]
    prediction_year = config.get("load_options")["prediction_year"]
    ssp = config.get("load_options")["ssp"]

    scenario_path = os.path.join(ssp_parentfolder, ssp)

    load_paths = []
    load_dir = os.path.join(
        ssp_parentfolder,
        str(ssp),
        str(prediction_year),
        "era5_" + str(weather_year),
    )

    file_names = []
    for continent in region_load:
        sel_ext = ".nc"
        for ext in [".nc", ".csv"]:
            load_path = os.path.join(BASE_DIR, str(load_dir), str(continent) + str(ext))
            if os.path.exists(load_path):
                sel_ext = ext
                break
        file_name = str(continent) + str(sel_ext)
        load_path = os.path.join(str(load_dir), file_name)
        load_paths.append(load_path)
        file_names.append(file_name)

    logger.info(
        f"Demand data folder: {load_dir}, load path is {load_paths}.\n"
        + f"Expected files: "
        + "; ".join(file_names)
    )

    return load_paths


def shapes_to_shapes(orig, dest):
    """
    Adopted from vresutils.transfer.Shapes2Shapes()
    """
    orig_prepped = list(map(prep, orig))
    transfer = sparse.lil_matrix((len(dest), len(orig)), dtype=float)

    for i, j in product(range(len(dest)), range(len(orig))):
        if orig_prepped[j].intersects(dest[i]):
            area = orig[j].intersection(dest[i]).area
            transfer[i, j] = area / dest[i].area

    return transfer


def custom_voronoi_partition_pts(points, outline, add_bounds_shape=True, multiplier=5):
    """
    Compute Voronoi polygons for points clipped to outline.

    This mirrors the bus-region helper but is local to demand building so the
    H2G demand allocation can create one demand cell for every AC bus without
    changing the global bus-region workflow.
    """
    from scipy.spatial import Voronoi

    points = np.asarray(points, dtype=float)

    if len(points) == 1:
        return [outline]

    xmin, ymin = np.amin(points, axis=0)
    xmax, ymax = np.amax(points, axis=0)

    if add_bounds_shape:
        minx_o, miny_o, maxx_o, maxy_o = outline.boundary.bounds
        xmin = min(xmin, minx_o)
        ymin = min(ymin, miny_o)
        xmax = min(xmax, maxx_o)
        ymax = min(ymax, maxy_o)

    xspan = xmax - xmin
    yspan = ymax - ymin
    if xspan <= 0.0 or yspan <= 0.0:
        raise ValueError("Cannot create Voronoi demand regions from collinear bounds.")

    vcells = Voronoi(
        np.vstack(
            (
                points,
                [
                    [xmin - multiplier * xspan, ymin - multiplier * yspan],
                    [xmin - multiplier * xspan, ymax + multiplier * yspan],
                    [xmax + multiplier * xspan, ymin - multiplier * yspan],
                    [xmax + multiplier * xspan, ymax + multiplier * yspan],
                ],
            )
        )
    )

    if not outline.is_valid:
        outline = outline.buffer(0)

    polygons_arr = np.empty((len(points),), "object")
    for i in range(len(points)):
        poly = Polygon(vcells.vertices[vcells.regions[vcells.point_region[i]]])
        if not poly.is_valid:
            poly = poly.buffer(0)
        polygons_arr[i] = poly.intersection(outline)

    return polygons_arr


def build_all_ac_bus_regions(n, admin_shapes, countries):
    """
    Create demand regions for all AC buses using Voronoi cells.
    """
    if "carrier" not in n.buses.columns:
        raise ValueError("Base network buses must define a carrier column.")
    if not {"x", "y", "country"}.issubset(n.buses.columns):
        raise ValueError("Base network buses must define x, y and country columns.")

    carrier = n.buses["carrier"].fillna("").astype(str).str.upper()
    xy = n.buses[["x", "y"]].apply(pd.to_numeric, errors="coerce")
    valid_xy = np.isfinite(xy["x"]) & np.isfinite(xy["y"])
    country_mask = n.buses["country"].astype(str).isin(countries)
    ac_buses = n.buses.loc[carrier.eq("AC") & country_mask & valid_xy, ["x", "y", "country"]]

    if ac_buses.empty:
        raise ValueError("No AC buses with valid coordinates found for demand allocation.")

    demand_regions = []
    for country in countries:
        onshore_locs = ac_buses.loc[ac_buses["country"] == country, ["x", "y"]]
        if onshore_locs.empty:
            logger.warning(f"No AC buses found for {country}.")
            continue

        shapes_country = admin_shapes.loc[admin_shapes.country == country]
        if shapes_country.empty:
            raise ValueError(f"No administrative shapes found for country {country}.")

        outline = make_valid(shapes_country.geometry.unary_union)
        if outline.is_empty:
            raise ValueError(f"Administrative outline for {country} is empty.")

        onshore_geometry = custom_voronoi_partition_pts(onshore_locs.values, outline)
        temp_region = gpd.GeoDataFrame(
            {
                "name": onshore_locs.index.astype(str),
                "x": onshore_locs["x"].to_numpy(),
                "y": onshore_locs["y"].to_numpy(),
                "country": country,
                "geometry": onshore_geometry,
            },
            crs=admin_shapes.crs,
        )

        invalid = temp_region[
            ~(temp_region.geometry.is_valid & ~temp_region.geometry.is_empty)
        ]
        if not invalid.empty:
            raise ValueError(
                "Failed to create valid AC-bus demand regions for: "
                + ", ".join(invalid["name"].astype(str))
            )
        demand_regions.append(temp_region)

    if not demand_regions:
        raise ValueError("No AC-bus demand regions could be created.")

    regions = gpd.GeoDataFrame(
        pd.concat(demand_regions, ignore_index=True),
        crs=admin_shapes.crs,
    ).set_index("name")
    logger.info(
        "Using all-AC-bus demand regions: %d AC buses receive demand profiles."
        % len(regions)
    )
    return regions


def load_demand_csv(path):
    df = read_csv_nafix(path, sep=";")
    df.time = pd.to_datetime(df.time, format="%Y-%m-%d %H:%M:%S")
    load_regions = {c: n for c, n in zip(df.region_code, df.region_name)}

    gegis_load = df.set_index(["region_code", "time"]).to_xarray()
    gegis_load = gegis_load.assign_coords(
        {
            "region_name": (
                "region_code",
                [name for (code, name) in load_regions.items()],
            )
        }
    )
    return gegis_load


def _resolve_optional_path(path):
    if not path:
        return ""
    return path if os.path.isabs(path) else os.path.join(BASE_DIR, path)


def _read_region_mapping(mapping_path, region_column, buses):
    if not mapping_path or not os.path.exists(mapping_path):
        raise FileNotFoundError(
            "Demand allocation bus-region mapping file not found: "
            f"{mapping_path}"
        )

    mapping = read_csv_nafix(mapping_path)
    required_cols = {"bus", region_column}
    if not required_cols.issubset(mapping.columns):
        raise ValueError(
            "Demand allocation bus-region mapping must contain columns "
            f"{required_cols}. Found {set(mapping.columns)}"
        )

    mapping = mapping.assign(
        bus=lambda df: df["bus"].astype(str).str.strip(),
        **{region_column: lambda df: df[region_column].astype(str).str.strip()},
    )
    if mapping["bus"].duplicated().any():
        duplicates = mapping.loc[mapping["bus"].duplicated(), "bus"].tolist()
        raise ValueError(
            "Demand allocation bus-region mapping contains duplicate buses: "
            + ", ".join(duplicates)
        )

    mapped_buses = pd.Index(mapping["bus"])
    missing_buses = buses.difference(mapped_buses)
    extra_buses = mapped_buses.difference(buses)
    if not missing_buses.empty:
        raise ValueError(
            "Demand allocation mapping missing demand buses: "
            + ", ".join(map(str, missing_buses))
        )
    if not extra_buses.empty:
        logger.info(
            "Demand allocation mapping contains non-demand buses that are ignored: "
            + ", ".join(map(str, extra_buses))
        )

    return mapping.set_index("bus").reindex(buses)[region_column]


def _read_target_region_shares(region_shares_config):
    if not isinstance(region_shares_config, dict) or not region_shares_config:
        raise ValueError(
            "load_options.demand_allocation.region_shares must be a non-empty mapping."
        )

    shares = pd.Series(region_shares_config, dtype=float)
    shares.index = pd.Index([str(i).strip() for i in shares.index])
    shares = shares.groupby(level=0).sum()

    if (shares < 0).any():
        raise ValueError("Demand allocation region_shares contain negative values.")

    total_share = float(shares.sum())
    if total_share <= 0.0:
        raise ValueError("Demand allocation region_shares sum to zero.")
    if not np.isclose(total_share, 1.0, atol=1e-6):
        logger.warning(
            "Demand allocation region_shares sum to %.6f and will be normalized.",
            total_share,
        )
    return shares / total_share


def _read_inline_access_shares(access_shares_config):
    if not isinstance(access_shares_config, dict) or not access_shares_config:
        raise ValueError(
            "load_options.demand_allocation.access_shares must be a non-empty mapping."
        )

    shares = pd.DataFrame.from_dict(access_shares_config, orient="index")
    shares.index = pd.Index([str(i).strip() for i in shares.index])
    required_cols = {"grid", "isolated", "standalone"}
    if not required_cols.issubset(shares.columns):
        raise ValueError(
            "Demand allocation access_shares must define grid, isolated and "
            f"standalone for each region. Found {set(shares.columns)}"
        )

    for col in ["grid", "isolated", "standalone"]:
        shares[col] = pd.to_numeric(shares[col], errors="raise")

    if (shares[["grid", "isolated", "standalone"]] < 0).any().any():
        raise ValueError("Demand allocation access_shares contain negative values.")

    share_sum = shares[["grid", "isolated", "standalone"]].sum(axis=1)
    if not np.allclose(share_sum, 1.0, atol=1e-6):
        invalid = share_sum[~np.isclose(share_sum, 1.0, atol=1e-6)]
        raise ValueError(
            "Demand allocation access_shares must sum to 1 per region. Invalid sums: "
            + invalid.to_dict().__repr__()
        )

    return shares[["grid", "isolated", "standalone"]]


def _write_optional_csv(df, path):
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def apply_demand_allocation(demand_profiles, allocation_config):
    """
    Apply regional demand targets and split demand into grid and offgrid demand.

    The input profiles already contain PyPSA-Earth's GDP/population bus allocation.
    This function preserves the relative PyPSA bus distribution within each custom
    region while enforcing configured regional demand shares.
    """
    if not allocation_config or not allocation_config.get("enabled", False):
        return demand_profiles

    gamma = float(allocation_config.get("gamma_industrial", 0.30))
    if gamma < 0.0 or gamma > 1.0:
        raise ValueError(
            "load_options.demand_allocation.gamma_industrial must be in [0, 1]."
        )

    mapping_path = _resolve_optional_path(
        allocation_config.get("bus_region_mapping", "")
    )
    region_column = str(allocation_config.get("region_column", "region")).strip()
    bus_shares_output = _resolve_optional_path(
        allocation_config.get("bus_shares_output", "")
    )
    offgrid_output = _resolve_optional_path(
        allocation_config.get("offgrid_demand_output", "")
    )

    buses = pd.Index(demand_profiles.columns.astype(str))
    demand_profiles = demand_profiles.copy()
    demand_profiles.columns = buses

    region_by_bus = _read_region_mapping(mapping_path, region_column, buses)
    target_region_shares = _read_target_region_shares(
        allocation_config.get("region_shares", {})
    )
    access_shares = _read_inline_access_shares(
        allocation_config.get("access_shares", {})
    )

    mapped_regions = pd.Index(region_by_bus.unique())
    missing_target_regions = mapped_regions.difference(target_region_shares.index)
    if not missing_target_regions.empty:
        raise ValueError(
            "Demand allocation mapping references regions without region_shares: "
            + ", ".join(map(str, missing_target_regions))
        )
    missing_access_regions = mapped_regions.difference(access_shares.index)
    if not missing_access_regions.empty:
        raise ValueError(
            "Demand allocation mapping references regions without access_shares: "
            + ", ".join(map(str, missing_access_regions))
        )

    unused_target_regions = target_region_shares.index.difference(mapped_regions)
    positive_unused = target_region_shares.loc[unused_target_regions]
    positive_unused = positive_unused[positive_unused > 0.0]
    if not positive_unused.empty:
        raise ValueError(
            "Demand allocation region_shares contain regions without demand buses: "
            + ", ".join(map(str, positive_unused.index))
        )

    total_by_bus = demand_profiles.sum(axis=0)
    total_demand = float(total_by_bus.sum())
    if total_demand <= 0.0:
        raise ValueError("Demand allocation cannot be applied to zero total demand.")

    pypsa_bus_shares = total_by_bus / total_demand
    pypsa_region_shares = pypsa_bus_shares.groupby(region_by_bus).sum()
    zero_regions = pypsa_region_shares[pypsa_region_shares <= 0.0].index
    if len(zero_regions):
        raise ValueError(
            "Demand allocation has regions with zero PyPSA demand share: "
            + ", ".join(map(str, zero_regions))
        )

    allocated_bus_shares = pypsa_bus_shares.copy()
    for bus in buses:
        region = region_by_bus.loc[bus]
        allocated_bus_shares.loc[bus] = (
            target_region_shares.loc[region]
            * pypsa_bus_shares.loc[bus]
            / pypsa_region_shares.loc[region]
        )
    allocated_bus_shares = allocated_bus_shares / allocated_bus_shares.sum()

    total_by_snapshot = demand_profiles.sum(axis=1)
    allocated_profiles = pd.DataFrame(
        total_by_snapshot.to_numpy()[:, None]
        * allocated_bus_shares.to_numpy()[None, :],
        index=demand_profiles.index,
        columns=buses,
    )

    grid_factors = (
        gamma
        + (1.0 - gamma)
        * access_shares.loc[region_by_bus, ["grid", "isolated"]]
        .sum(axis=1)
        .to_numpy()
    )
    offgrid_factors = (
        (1.0 - gamma) * access_shares.loc[region_by_bus, "standalone"].to_numpy()
    )
    grid_factors = pd.Series(grid_factors, index=buses)
    offgrid_factors = pd.Series(offgrid_factors, index=buses)

    grid_profiles = allocated_profiles.mul(grid_factors, axis=1)
    offgrid_profiles = allocated_profiles.mul(offgrid_factors, axis=1)

    if not np.allclose(allocated_profiles, grid_profiles + offgrid_profiles, atol=1e-6):
        raise ValueError("Demand allocation split does not preserve raw bus demand.")

    if offgrid_output:
        os.makedirs(os.path.dirname(offgrid_output), exist_ok=True)
        offgrid_profiles.to_csv(offgrid_output, header=True)
        logger.info(f"Offgrid demand profiles written to {offgrid_output}.")

    bus_share_table = pd.DataFrame(
        {
            "bus": buses,
            "region": region_by_bus.to_numpy(),
            "pypsa_share": pypsa_bus_shares.reindex(buses).to_numpy(),
            "share": allocated_bus_shares.reindex(buses).to_numpy(),
            "grid_factor": grid_factors.reindex(buses).to_numpy(),
            "offgrid_factor": offgrid_factors.reindex(buses).to_numpy(),
        }
    )
    _write_optional_csv(bus_share_table, bus_shares_output)

    logger.info(
        "Applied demand allocation: raw %.3f TWh, grid %.3f TWh, offgrid %.3f TWh."
        % (
            allocated_profiles.sum().sum() / 1e6,
            grid_profiles.sum().sum() / 1e6,
            offgrid_profiles.sum().sum() / 1e6,
        )
    )
    return grid_profiles


def build_demand_profiles(
    n,
    load_paths,
    regions,
    admin_shapes,
    countries,
    scale,
    start_date,
    end_date,
    out_path,
    demand_allocation_config=None,
):
    """
    Create csv file of electric demand time series.

    Parameters
    ----------
    n : pypsa network
    load_paths: paths of the load files
    regions : .geojson
        Contains bus_id of low voltage substations and
        bus region shapes (voronoi cells)
    admin_shapes : .geojson
        contains subregional gdp, population and shape data
    countries : list
        List of countries that is config input
    scale : float
        The scale factor is multiplied with the load (1.3 = 30% more load)
    start_date: parameter
        The start_date is the first hour of the first day of the snapshots
    end_date: parameter
        The end_date is the last hour of the last day of the snapshots

    Returns
    -------
    demand_profiles.csv : csv file containing the electric demand time series
    """
    shapes = gpd.read_file(admin_shapes).set_index("GADM_ID")
    shapes["geometry"] = shapes["geometry"].apply(lambda x: make_valid(x))

    use_all_ac_buses = bool(
        demand_allocation_config
        and demand_allocation_config.get("use_all_ac_buses", False)
    )
    if use_all_ac_buses:
        regions = build_all_ac_bus_regions(n, shapes, countries)
    else:
        substation_lv_i = n.buses.index[n.buses["substation_lv"]]
        regions = gpd.read_file(regions).set_index("name").reindex(substation_lv_i)
    load_paths = load_paths

    gegis_load_list = []

    for path in load_paths:
        if str(path).endswith(".csv"):
            gegis_load_xr = load_demand_csv(path)
        else:
            # Merge load .nc files: https://stackoverflow.com/questions/47226429/join-merge-multiple-netcdf-files-using-xarray
            gegis_load_xr = xr.open_mfdataset(path, combine="nested")
        gegis_load_list.append(gegis_load_xr)

    logger.info(f"Merging demand data from paths {load_paths} into the load data frame")
    gegis_load = xr.merge(gegis_load_list)
    gegis_load = gegis_load.to_dataframe().reset_index().set_index("time")

    # filter load for analysed countries
    gegis_load = gegis_load.loc[gegis_load.region_code.isin(countries)]

    if isinstance(scale, dict):
        logger.info(f"Using custom scaling factor for load data.")
        DEFAULT_VAL = scale.get("DEFAULT", 1.0)
        for country in countries:
            scale.setdefault(country, DEFAULT_VAL)

        for country, scale_country in scale.items():
            gegis_load.loc[
                gegis_load.region_code == country, "Electricity demand"
            ] *= scale_country

    elif isinstance(scale, (int, float)):
        logger.info(f"Load data scaled with scaling factor {scale}.")
        gegis_load["Electricity demand"] *= scale

    def upsample(cntry, group):
        """
        Distributes load in country according to population and gdp.
        """
        l = gegis_load.loc[gegis_load.region_code == cntry]["Electricity demand"]
        if len(group) == 1:
            return pd.DataFrame({group.index[0]: l})
        else:
            shapes_cntry = shapes.loc[shapes.country == cntry]
            transfer = shapes_to_shapes(group, shapes_cntry.geometry).T.tocsr()
            gdp_n = pd.Series(
                transfer.dot(shapes_cntry["gdp"].fillna(1.0).values), index=group.index
            )
            pop_n = pd.Series(
                transfer.dot(shapes_cntry["pop"].fillna(1.0).values), index=group.index
            )

            # relative factors 0.6 and 0.4 have been determined from a linear
            # regression on the country to EU continent load data
            # (refer to vresutils.load._upsampling_weights)
            # TODO: require adjustment for Africa
            factors = normed(0.6 * normed(gdp_n) + 0.4 * normed(pop_n))
            if factors.sum() == 0:
                logger.warning(
                    f"Upsampling factors for {cntry} are all zero, returning uniform distribution across {len(factors)} shapes."
                )
                factors = pd.Series(
                    np.ones(len(factors)) / len(factors), index=factors.index
                )
            return pd.DataFrame(
                factors.values * l.values[:, np.newaxis],
                index=l.index,
                columns=factors.index,
            )

    demand_profiles = pd.concat(
        [
            upsample(cntry, group)
            for cntry, group in regions.geometry.groupby(regions.country)
        ],
        axis=1,
    )

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date) - pd.Timedelta(hours=1)
    demand_profiles = demand_profiles.loc[start_date:end_date]

    if demand_allocation_config and demand_allocation_config.get("enabled", False):
        demand_profiles = apply_demand_allocation(
            demand_profiles, demand_allocation_config
        )
        demand_profiles.to_csv(out_path, header=True)
        logger.info("Demand_profiles csv file created for the corresponding snapshots.")
        return

    demand_profiles.to_csv(out_path, header=True)

    logger.info(f"Demand_profiles csv file created for the corresponding snapshots.")


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("build_demand_profiles")

    configure_logging(snakemake)

    n = pypsa.Network(snakemake.input.base_network)

    # Snakemake imports:
    regions = snakemake.input.regions
    load_paths = snakemake.input["load"]
    countries = snakemake.params.countries
    admin_shapes = snakemake.input.gadm_shapes
    scale = snakemake.params.load_options.get("scale", 1.0)
    start_date = snakemake.params.snapshots["start"]
    end_date = snakemake.params.snapshots["end"]
    demand_allocation_config = snakemake.params.load_options.get(
        "demand_allocation", {}
    )
    out_path = snakemake.output[0]

    build_demand_profiles(
        n,
        load_paths,
        regions,
        admin_shapes,
        countries,
        scale,
        start_date,
        end_date,
        out_path,
        demand_allocation_config,
    )
