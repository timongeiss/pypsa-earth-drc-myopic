<!--
SPDX-FileCopyrightText:  PyPSA-Earth and PyPSA-Eur Authors
SPDX-FileCopyrightText:  2026 Timon Geiss, Anton Achhammer, Alexander Meisinger, Leon Schumm, Michael Sterner

SPDX-License-Identifier: AGPL-3.0-or-later
-->

# DRC Myopic PyPSA-Earth Energy System Model

This repository contains a PyPSA-Earth based model setup for the Democratic
Republic of the Congo (DRC). It accompanies the paper:

> Timon Geiss, Anton Achhammer, Alexander Meisinger, Leon Schumm and Michael
> Sterner, "A Constrained PyPSA-Earth-Based Energy System Model for the
> Democratic Republic of the Congo: Implications for Hydrogen Export from
> Hydropower under Real-World Conditions".

The model studies a constrained, sequential pathway for the DRC power and
hydrogen export system. It builds on PyPSA-Earth and adds DRC-specific custom
network data, hydro assumptions, demand allocation, hydrogen export scenarios
and myopic brownfield capacity transfer.

## Model Scope

The main scenario is defined in
`configs/scenarios_H2G/config.H2G_A_CD_myopic.yaml`.

Key settings:

- Country: Democratic Republic of the Congo (`CD`)
- Foresight mode: `myopic`
- Planning horizons: `2025`, `2035`, `2050`
- Temporal resolution: `1H`
- Weather year: `2013`
- Network resolution: custom all-node DRC network (`clusters: all`)
- Constraint option: country/carrier capacity limits (`opts: CCL`)
- Solver in the paper config: `gurobi`
- Additional 2035 sensitivity: `early_large`, allowing Grand Inga and Pioka as
  2035 hydro candidates while keeping hydrogen export demand at `0` TWh/a
- Hydrogen export cases in 2050: `no_large_hydro`, `0`, `23.33`, `78.33`,
  `133.32` TWh per year

The `no_large_hydro` case is a reference branch that removes selected large
2050 hydro candidates. The other export cases represent increasing hydrogen
export demand while retaining the same sequential pathway structure.

## Repository Layout

- `configs/scenarios_H2G/config.H2G_A_CD_myopic.yaml`
  Main DRC myopic scenario configuration.
- `data/custom/drc_myopic/`
  Custom DRC input data for demand regions, year-specific base networks,
  custom substations and the 2025 custom solar profile.
- `data/custom/export_ports.csv`
  Custom export-port definition used for hydrogen export cases.
- `data/custom_powerplants.csv`
  Custom power plant and hydro candidate assumptions used by the model.
- `hybrid_results_myopic.py`
  Plotting and result-inspection script for solved DRC myopic networks.
- `A_Constrained_PyPSA_Earth_Based_Energy_System_Model_for_the_Democratic_Republic_of_the_Congo.pdf`
  Local manuscript PDF used for the repository citation metadata.

Generated workflow outputs are written below `resources/`, `networks/`,
`results/`, `logs/` and `benchmarks/`. These folders are not intended to be
treated as source files.

## Data Requirements

This scenario uses `retrieve_databundle: false` and `build_cutout: false`.
Therefore, a run expects several PyPSA-Earth baseline datasets to be present
locally, including:

- `data/natura/natura.tiff`
- `data/eez/eez_v11.gpkg`
- `data/GDP/GDP_PPP_1990_2015_5arcmin_v2.nc`
- `data/hydrobasins/hybas_world.shp` and its shapefile sidecars
- `data/ssp2-2.6/.../Africa.nc`
- `data/copernicus/PROBAV_LC100_global_v3.0.1_2019-nrt_Discrete-Classification-map_EPSG-4326.tif`
- `data/gebco/GEBCO_2025_sub_ice.nc`
- `cutouts/cutout-2013-era5.nc`

Large baseline datasets should normally be restored from the PyPSA-Earth data
bundle, a shared local data directory or an archived publication dataset rather
than committed to Git.

## Running

Activate the PyPSA-Earth environment from the repository root:

```bash
conda activate pypsa-earth
```

Run a dry run first:

```bash
snakemake solve_sector_networks --cores 4 --configfile configs/scenarios_H2G/config.H2G_A_CD_myopic.yaml --dry-run
```

Start the full model run:

```bash
snakemake solve_sector_networks --cores 4 --configfile configs/scenarios_H2G/config.H2G_A_CD_myopic.yaml
```

Snakemake will resume from completed outputs if the workflow stops after a
successful subset of jobs.

## Citation

If you use this repository, cite the accompanying paper and this repository.
Repository-level citation metadata are provided in `CITATION.cff`.

The model is derived from PyPSA-Earth. Please also cite the relevant PyPSA-Earth
publications:

- Maximilian Parzen et al., "PyPSA-Earth: A new global open energy system
  optimization model demonstrated in Africa", Applied Energy, 341, 2023,
  https://doi.org/10.1016/j.apenergy.2023.121096
- Hazem Abdel-Khalek et al., "PyPSA-Earth sector-coupled: A global open-source
  multi-energy system model showcased for hydrogen applications in countries of
  the Global South", Applied Energy, 383, 2025,
  https://doi.org/10.1016/j.apenergy.2025.125316

## License

This repository inherits the license structure of the original PyPSA-Earth
project.

- Source code and repository metadata: `AGPL-3.0-or-later`
- Data and documentation where annotated by `REUSE.toml`: `CC-BY-4.0`
- Public-domain style project metadata where annotated: `CC0-1.0`

The license texts are kept in `LICENSES/` and the REUSE annotations are kept in
`REUSE.toml`.
