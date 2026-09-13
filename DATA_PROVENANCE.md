<!--
SPDX-FileCopyrightText: 2026 Timon Geiss

SPDX-License-Identifier: CC-BY-4.0
-->

# Data Provenance

This document separates authorship of the study-specific compilation and
transformations from rights in the underlying source material. A CC-BY-4.0
annotation for a study-authored input applies only where indicated; it does not
replace third-party attribution or licence requirements.

The comparison against upstream refers to PyPSA-Earth commit
`9ff3ee2f7fb69e911d82b5abd7395d7b29f0a357`.

| File / dataset | Purpose | Compiled by | Main underlying sources | Transformation | Redistribution/licence notes |
| --- | --- | --- | --- | --- | --- |
| `data/custom_powerplants.csv` | Existing plants, hydro candidates and modelled 2025 oil generators | Timon Geiss | Literature and public project-level sources documented in the accompanying manuscript and Supplementary Material; exact record-level mapping **to be confirmed** | Study-specific validation, selection, additions, classifications and modelling annotations in the PyPSA-Earth custom-powerplant schema. Obsolete fixed 2035 oil carryover rows were removed after the workflow audit below. | PyPSA-Earth attribution is preserved for the upstream file/schema. CC-BY-4.0 covers the study-specific compilation and transformations only; source material remains subject to its own terms. |
| `data/custom/export_ports.csv` | Six hydrogen-export nodes with explicit model-bus assignments | Timon Geiss | The year-specific study bus tables; the underlying source of those bus coordinates is covered by the base-network entries below | Selected six study buses and copied their exact coordinates into the port table | In the current implementation, the `bus` column controls the export connection; `x` and `y` are not used when that column is present. No separate GeoNames data were introduced. Preserve PyPSA-Earth attribution for the upstream file/schema. |
| `data/custom/pipelines.csv` | Explicitly empty custom gas-network input | Timon Geiss | No substantive source records | Created a header-only control input with the columns expected by the workflow | CC-BY-4.0 covers the study-specific empty input; there are no pipeline observations to attribute. |
| `data/custom/drc_myopic/base_network/*/custom_all_buses_build_network.csv` | Year-specific network buses and coordinates | Timon Geiss | SNEL planning information compiled by Maupin (2017), study assumptions, and a small coordinate subset matching the retained PyPSA-Earth OpenStreetMap substation extract | Manual reconstruction, selection and planning-horizon assignment. AC voltages are standardized to 220 kV and DC voltages to 500 kV; these values were not copied from the matched OSM substations. At CSV precision, the coordinates of Gbadolite, Kindu, Kisangani and Mbandaka match OSM substation centroids; Mbandaka is numerically identical before rounding. | Only those four coordinate pairs are OSM-derived; the remaining modelling choices and transformations are study-specific. The complete CSV files are nevertheless treated conservatively under ODbL-1.0 with OpenStreetMap contributors and Timon Geiss attribution, consistently with the line CSV treatment. Exact non-OSM source records for the remaining coordinates are **to be confirmed**. |
| `data/custom/drc_myopic/base_network/*/custom_all_converters_build_network.csv` and `custom_all_transformers_build_network.csv` | Converter connections and empty transformer tables | Timon Geiss | Study network topology and modelling assumptions; exact supporting source records **to be confirmed** | Created converter links between the study AC/DC buses; retained header-only transformer tables | CC-BY-4.0 covers the study-specific tables. |
| `data/custom/drc_myopic/base_network/*/custom_all_lines_build_network.csv` | Existing and planned AC/DC transmission topology for 2025, 2035 and 2050 | Timon Geiss | SNEL planning information compiled by Maupin (2017), OpenStreetMap line records, and study assumptions | Twelve recurring numeric line rows can be matched to the retained PyPSA-Earth OSM extract. Their OSM fields include the line identifier and source values for `circuits`, `voltage` and `dc`. The study changes `1064077743` from 70 kV to 220 kV; changes circuit counts for `194187707` (1 to 3), `384190117` (2 to 4), and five HVDC records (1/3 to 2/3); splits OSM way `224147292` into two model segments; assigns study bus endpoints; and adds manually reconstructed/planned lines. | Conservatively treated as an OSM-derived database under ODbL-1.0, with “OpenStreetMap contributors” attribution. The study selection and additions are identified separately as Timon Geiss's work. |
| `data/custom/drc_myopic/demand_regions_2025.csv`, `demand_regions_2035.csv`, `demand_regions_2050.csv` | Map model buses to study demand regions | Timon Geiss | World Bank methodology used as the study basis; exact publication/dataset **to be confirmed** | Manual study-specific regional assignment for each planning horizon | CC-BY-4.0 covers the study-specific assignment; confirm the exact World Bank citation and any required attribution. |
| `scripts/scale_solar_profile.py` → local `data/custom/drc_myopic/renewable_profiles/solar_custom_2025.nc` | Creates the required 2025 solar-capacity input locally | Timon Geiss (scaling script); PyPSA-Earth workflow for the source profile | Standard PyPSA-Earth `build_renewable_profiles`; an ERA5 2013 cutout through atlite; Copernicus PROBA-V land cover; Natura 2000 exclusion raster; and study bus-region shapes | Loads the standard `profile_solar.nc`, multiplies only `p_nom_max` by one uniform factor so that its total changes from approximately 1,036,877.229624 MW to exactly 250 MW, and writes the custom file. `profile`, `weight`, `potential`, coordinates, buses and all other variables remain unchanged. | The transformation script is distributed under AGPL-3.0-or-later. The generated NetCDF is required locally but is ignored and not redistributed, avoiding a repository-level licence assertion for its unchanged third-party-derived contents. |
| `data/agg_p_nom_minmax.csv` | Header-only input required by the configured `CCL` option | Timon Geiss (modification); PyPSA-Earth authors (upstream template) | PyPSA-Earth upstream template | Added the planning-horizon columns. Three wind maxima were removed after verifying that wind is non-extendable in every published horizon and that the CCL constraint filters to extendable carriers only. | PyPSA-Earth and Timon Geiss attribution; CC-BY-4.0. No numerical capacity limit remains in this file. |

## OpenStreetMap field audit

The OSM-derived subset of each line table consists of these recurring source
identifiers:

`1028776989`, `1064077743`, `194187707`, `286737300`,
`384190117`, `865689670`, `224147292`, `460419487`,
`626214770`, `778351007` and `979570584`.

The remaining `figure_*` and `planned_*` line identifiers are
study-authored reconstruction/planning records. Assigning ODbL-1.0 to the whole
line CSV is the conservative publication treatment because OSM-derived and
study-authored records form one combined network table. [OpenStreetMap's
copyright page](https://www.openstreetmap.org/copyright) requests the
attribution “© OpenStreetMap contributors” and identifies its database as
available under ODbL.

## Bus coordinate and voltage audit

The bus tables were compared with the locally retained PyPSA-Earth OSM clean
substation extract. Only Mbandaka has an exactly identical unrounded coordinate
pair. Gbadolite, Kindu and Kisangani match the corresponding OSM centroids when
rounded to the four decimal places used in the study CSVs. Other bus locations
do not match an OSM substation at that precision.

None of these four matched OSM substations supplies the voltage used in the bus
tables: their OSM voltages are 132 kV or 70 kV, whereas the study tables use
220 kV for every AC bus and 500 kV for DC buses. The voltage fields are
study modelling assumptions rather than direct OSM copies.

## Solar profile audit

The retained source profile was:

`/home/get39559/pypsa-earth/resources/H2G_A_CD_2025/renewable_profiles/profile_solar.nc`

That local path documents the audit trail but is not a required public-repo
path. Its SHA-256 is
`975d91d599ed3e8ffa3a9558e169e2c4feb04fd485f9dd25f2ad42bd2065212b`.
The formerly retained custom reference file's SHA-256 was
`af4c71763a2e8f2cfc4b339323638f04e22edbc16efe5e7cf53df857c5fee81b`.

The source rule uses ERA5/atlite for the hourly solar profile and uses
Copernicus land cover, Natura exclusions and bus-region geometry to calculate
land availability and `p_nom_max`. The [Climate Data Store ERA5
entry](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels)
currently identifies ERA5 under a CC-BY licence. GEBCO is a generic rule input but is not
used by the configured solar technology because no solar depth restriction is
set. No claim is made here that the study transformation supersedes licences
attached to these inputs. The binary snapshot is no longer redistributed;
`scripts/scale_solar_profile.py` now records and performs the transformation.

## Oil carryover audit

Twenty rows formerly present in `data/custom_powerplants.csv` were labelled
`generated_2025_oil_carryover_2035`. They had been produced by the older local script
`/home/get39559/pypsa-earth/scripts/update_2035_oil_from_2025.py`
from a solved 2025 network and total approximately 195.691 MW.

The rows entered the 2035 electricity-only network as fixed generators, but
`prepare_sector_network.py` removes electricity-only oil generators and rebuilds
oil conversion links. The subsequent brownfield step replaces these with the
links carried from the solved 2025 network. Consequently, none of the C128–C147
carryover generators is present in either audited 2035 solved network.

The final 2035 zero-export result produces 230.894 GWh of electricity from its
2025-carried oil links (approximately 0.58% of AC electricity demand); the
`early_large` result produces 60.507 GWh (approximately 0.15%). That dispatch
comes from the normal myopic 2025 brownfield transfer, not from the obsolete
195.691 MW input rows. The obsolete rows were therefore removed from the
publication input.

## Inputs intentionally not redistributed

The two former `custom_substations_2025.geojson` and
`custom_substations_2035.geojson` files were removed from the publication
working tree. The configured workflow reads the year-specific base-network
CSVs directly, so those GeoJSON rebuild artefacts were not required to
reproduce the published path. Removing them also avoids redistributing the
least-resolved OSM/Bing provenance. They remain recoverable from Git history.

The generated `solar_custom_2025.nc` is likewise not redistributed. It is
created locally by the documented Snakemake target and is explicitly ignored
by Git.

## Evidence retained in the audited files

- The DRC input set entered repository history in commit
  `6482bd4f2adae14b96d8823594e9c21a845c6bb1`; no earlier
  source-preparation history is present for the new files.
- The export buses are selected explicitly in `scripts/add_export.py` when a
  `bus` column exists; coordinate-based GADM lookup is only the fallback.
