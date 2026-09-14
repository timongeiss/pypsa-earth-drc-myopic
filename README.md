<!--
SPDX-FileCopyrightText:  PyPSA-Earth and PyPSA-Eur Authors
SPDX-FileCopyrightText:  2026 Timon Geiss

SPDX-License-Identifier: AGPL-3.0-or-later
-->

# DRC Myopic PyPSA-Earth Energy System Model

This repository contains the reproducible implementation of a model setup for
the Democratic Republic of the Congo (DRC) accompanying the paper:

> Timon Geiss, Anton Achhammer, Alexander Meisinger, Leon Schumm and Michael
> Sterner, "A Constrained PyPSA-Earth-Based Energy System Model for the
> Democratic Republic of the Congo: Implications for Hydrogen Export from
> Hydropower under Real-World Conditions".

The model studies a constrained, sequential pathway for the DRC power and
hydrogen export system. It is a modified version of the
[PyPSA-Earth project](https://github.com/pypsa-meets-earth/pypsa-earth), based
on PyPSA-Earth commit
[`9ff3ee2f7fb69e911d82b5abd7395d7b29f0a357`](https://github.com/pypsa-meets-earth/pypsa-earth/commit/9ff3ee2f7fb69e911d82b5abd7395d7b29f0a357).
This repository is not an official PyPSA-Earth release and is not intended to
track or merge newer upstream versions. It adds DRC-specific custom network
data, hydro assumptions, demand allocation, hydrogen export scenarios and
myopic brownfield capacity transfer.

The reproducible version accompanying the paper is maintained on the
`paper-drc-myopic` branch, which is the default branch of this repository.
The archived paper release is identified by the tag `v1.0-paper`.

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
  Custom DRC input data for demand regions and year-specific base networks.
- `scripts/scale_solar_profile.py`
  Reproducibly creates the local 2025 DRC solar profile by scaling only
  `p_nom_max` in the standard PyPSA-Earth solar profile to 250 MW.
- `data/custom/export_ports.csv`
  Custom export-port definition used for hydrogen export cases.
- `data/custom_powerplants.csv`
  Custom power plant and hydro candidate assumptions used by the model.
- `data/custom/pipelines.csv`
  Empty custom-pipeline definition used to avoid substituting an external gas
  network where no study pipeline is specified.
- `data/agg_p_nom_minmax.csv`
  Country/carrier capacity limits used by the `CCL` scenario option.
- `hybrid_results_myopic.py`
  Plotting and result-inspection script for solved DRC myopic networks.
- `DATA_PROVENANCE.md`
  Provenance status and outstanding source/licence checks for custom inputs.

Generated workflow outputs are written below `resources/`, `networks/`,
`results/`, `logs/` and `benchmarks/`. These folders are not intended to be
treated as source files.

## Data Requirements

All small primary study-specific model inputs required by the DRC configuration
are included in this repository. The derived binary file
`data/custom/drc_myopic/renewable_profiles/solar_custom_2025.nc` is deliberately
not redistributed. It is generated locally from the standard PyPSA-Earth solar
profile by `scripts/scale_solar_profile.py`. Provenance and outstanding
source/licence checks are documented in
[`DATA_PROVENANCE.md`](DATA_PROVENANCE.md).

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

## Installation and Running

Clone the repository and create the Conda environment:

```bash
git clone https://github.com/timongeiss/pypsa-earth-drc-myopic.git
cd pypsa-earth-drc-myopic
conda env create --file envs/environment.yaml
```

Activate the PyPSA-Earth environment from the repository root:

```bash
conda activate pypsa-earth
```

Before the model run, create the non-redistributed custom solar NetCDF. This
rule first builds the standard 2025 PyPSA-Earth `profile_solar.nc` when needed,
then invokes `scripts/scale_solar_profile.py` and writes
`data/custom/drc_myopic/renewable_profiles/solar_custom_2025.nc`:

```bash
snakemake build_drc_solar_profile_2025 --cores 4 --configfile configs/scenarios_H2G/config.H2G_A_CD_myopic.yaml
```

The generated file is ignored by Git. The full workflow also knows this
dependency, but running the preparation target explicitly makes the required
transformation and its successful completion visible before optimization.

Run a dry run first:

```bash
snakemake solve_sector_networks_myopic --cores 4 --configfile configs/scenarios_H2G/config.H2G_A_CD_myopic.yaml --dry-run
```

Start the full model run:

```bash
snakemake solve_sector_networks_myopic --cores 4 --configfile configs/scenarios_H2G/config.H2G_A_CD_myopic.yaml
```

Snakemake will resume from completed outputs if the workflow stops after a
successful subset of jobs. The paper configuration selects Gurobi, so a working
Gurobi installation and licence are required. Platform-specific lock files are
available under `envs/` for recording the tested dependency resolution.

After the required solved networks are available, regenerate the study figures
and derived tables with:

```bash
python hybrid_results_myopic.py
```

## Data and Results Availability

This Git repository contains the code, configuration files, and small custom model inputs used in this study. Solved networks, intermediate resources, logs, benchmarks, and generated figures are intentionally excluded from version control.

The publication result artefacts, including solved NetCDF networks, comparison tables, and generated figures, are archived separately on Zenodo:

**Results dataset:** https://doi.org/10.5281/zenodo.22734263


## Citation

If you use this repository, cite the accompanying paper and this repository.
Repository-level citation metadata are provided in `CITATION.cff`.

The model is derived from
[PyPSA-Earth](https://github.com/pypsa-meets-earth/pypsa-earth). Please also
cite the relevant PyPSA-Earth publications:

- Maximilian Parzen et al., "PyPSA-Earth: A new global open energy system
  optimization model demonstrated in Africa", Applied Energy, 341, 2023,
  https://doi.org/10.1016/j.apenergy.2023.121096
- Hazem Abdel-Khalek et al., "PyPSA-Earth sector-coupled: A global open-source
  multi-energy system model showcased for hydrogen applications in countries of
  the Global South", Applied Energy, 383, 2025,
  https://doi.org/10.1016/j.apenergy.2025.125316

## License

This repository is derived from PyPSA-Earth and preserves the applicable
upstream licences and copyright notices.

- Source code derived from or extending PyPSA-Earth is distributed under
  `AGPL-3.0-or-later` where indicated.
- Study-authored model-input datasets are distributed under `CC-BY-4.0` where
  indicated.
- Inputs containing or derived from third-party data retain the applicable
  source licences and attribution requirements.
- OpenStreetMap-derived data are attributed to OpenStreetMap contributors and
  are subject to the Open Database License (`ODbL-1.0`) where applicable.
- `scripts/build_offgrid_myopic.py` retains Mohamed Amine Chebaane's
  attribution and its indicated `CC-BY-4.0` licence; the repository copy also
  contains a small DRC-side change to command-line list handling.
- File-specific copyright, licensing and data provenance are documented in
  `REUSE.toml` and `DATA_PROVENANCE.md`.

Large baseline datasets required by the standard PyPSA-Earth workflow are not
redistributed in this repository and must be obtained through the corresponding
PyPSA-Earth data-retrieval workflow or from their original providers. Licence
texts are kept in `LICENSES/`.
