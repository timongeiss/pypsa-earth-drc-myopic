<!--
SPDX-FileCopyrightText: 2026 Timon Geiss

SPDX-License-Identifier: CC-BY-4.0
-->

# Data Provenance

This document describes the provenance, study-specific transformations and
redistribution treatment of the custom model inputs included in this repository.

The DRC-specific datasets were compiled and adapted for the accompanying study.
Where third-party information is incorporated, the corresponding source rights
and attribution requirements remain unaffected by the study-specific
compilation or transformation.

The repository is based on PyPSA-Earth commit
`9ff3ee2f7fb69e911d82b5abd7395d7b29f0a357`.

| File / dataset | Purpose | Provenance and study-specific transformation | Redistribution / licence treatment |
| --- | --- | --- | --- |
| `data/custom_powerplants.csv` | Existing generation assets, future hydropower candidates and modelled 2025 oil generators | Compiled by Timon Geiss from literature and public project-level sources documented in the accompanying manuscript and Supplementary Material. Existing hydropower plants were validated primarily against OpenInfraMap and Wikipedia, with project-specific additions and corrections. Cascaded future hydropower potentials are based primarily on Global Energy Interconnection Development and Cooperation Organization (2020), while individually reported projects use the corresponding project-level sources documented in the study. Identified oil-fired plants are based on OpenInfraMap. Additional 1 MW backup generators at otherwise unrepresented AC buses are a study-specific modelling assumption motivated by the widespread use of private backup generation reported by World Bank (2020). Technology classifications, scenario availability, bus assignment and other modelling annotations are study-specific. A separate record-level source mapping for every CSV row was not retained; source provenance is documented at dataset and project level in the manuscript and Supplementary Material. | The PyPSA-Earth attribution for the upstream file/schema is preserved. CC-BY-4.0 applies to the study-specific compilation and transformations; underlying third-party source material remains subject to its respective terms. |
| `data/custom/export_ports.csv` | Six hydrogen-export nodes with explicit model-bus assignments | Compiled by Timon Geiss from the year-specific study bus tables. The six export locations are selected study nodes and their coordinates are copied from the corresponding model buses. In the implemented workflow, the explicit `bus` field determines the model connection; coordinate-based assignment is used only as a fallback. | The PyPSA-Earth attribution for the upstream file/schema is preserved. The study-specific compilation is distributed under CC-BY-4.0. Coordinate provenance follows that of the corresponding bus tables. |
| `data/custom/pipelines.csv` | Empty custom gas-network input | Study-specific header-only input created by Timon Geiss to prevent the workflow from introducing external pipeline records when no custom gas network is specified. | CC-BY-4.0. The file contains no substantive third-party observations. |
| `data/custom/drc_myopic/base_network/*/custom_all_buses_build_network.csv` | Year-specific model buses and node coordinates | Manually reconstructed by Timon Geiss as part of the study-specific DRC network representation, based primarily on SNEL planning information compiled by Maupin (2017), supplemented by study assumptions and selected OpenStreetMap-derived coordinates. At the precision used in the CSV files, the coordinates of Gbadolite, Kindu, Kisangani and Mbandaka match the corresponding OpenStreetMap substation locations. The remaining node coordinates form part of the study-specific spatial representation; a separate point-source mapping for each non-OSM coordinate was not retained. AC buses are assigned 220 kV and DC buses 500 kV as modelling assumptions; these voltage levels were not copied from the matched OSM substations. | Because the published tables combine study-authored content with OpenStreetMap-derived coordinate information, the complete bus CSV files are conservatively distributed under ODbL-1.0 with attribution to OpenStreetMap contributors and Timon Geiss. The model-node coordinates should not be interpreted as an authoritative geospatial inventory of physical substations. |
| `data/custom/drc_myopic/base_network/*/custom_all_lines_build_network.csv` | Existing and planned AC/DC transmission topology for 2025, 2035 and 2050 | Reconstructed by Timon Geiss from SNEL planning information compiled by Maupin (2017), OpenStreetMap line records and study assumptions. A subset of existing lines can be traced to OpenStreetMap way identifiers and associated source values for fields including `circuits`, `voltage` and `dc`. The study modifies selected voltage and circuit assumptions, assigns study-specific bus endpoints, splits selected source geometries into model segments, and adds manually reconstructed or planned corridors. | Conservatively distributed under ODbL-1.0 with attribution to OpenStreetMap contributors and Timon Geiss because OSM-derived and study-authored records form a combined network table. |
| `data/custom/drc_myopic/base_network/*/custom_all_converters_build_network.csv` and `custom_all_transformers_build_network.csv` | Converter connections and transformer-table inputs | Study-authored model-topology inputs created by Timon Geiss from the reconstructed DRC network. Converter records connect the study-specific AC and DC nodes required by the model. Transformer tables are intentionally header-only because no separate transformer representation is used in the published setup. | CC-BY-4.0. |
| `data/custom/drc_myopic/demand_regions_2025.csv`, `demand_regions_2035.csv`, `demand_regions_2050.csv` | Assignment of model buses to demand-allocation regions | Compiled by Timon Geiss. The regional structure follows World Bank (2020), *Increasing Access to Electricity in the Democratic Republic of Congo: Opportunities and Challenges*, which distinguishes the South-West, East and North-Center regions and provides the regional electricity-access structure used in the study. Assignment of individual model buses to these regions and the 75% / 15% / 10% allocation of national demand across the three regions are study-specific modelling choices. | CC-BY-4.0 applies to the study-specific bus-to-region mapping and modelling representation. |
| `scripts/scale_solar_profile.py` → local `data/custom/drc_myopic/renewable_profiles/solar_custom_2025.nc` | Creation of the REF25 solar-capacity input | The scaling script was written by Timon Geiss. It uses the standard 2025 solar renewable profile produced by the PyPSA-Earth `build_renewable_profiles` workflow. The source profile is based on a 2013 ERA5 cutout processed through atlite together with the land-availability and spatial inputs used by PyPSA-Earth, including Copernicus PROBA-V land cover, Natura exclusions and model bus regions. The script changes only `p_nom_max`, applying one uniform scaling factor so that the total installable PV capacity is reduced from approximately 1,036,877 MW to exactly 250 MW. The hourly profile, potential, weights, coordinates and other variables are preserved. | `scripts/scale_solar_profile.py` is distributed under AGPL-3.0-or-later. The generated NetCDF file is not redistributed and is ignored by Git. Users generate it locally from the standard PyPSA-Earth renewable-profile workflow, avoiding a separate repository-level licence assertion for the unchanged third-party-derived profile contents. |
| `data/agg_p_nom_minmax.csv` | Input required by the configured `CCL` option | Based on the PyPSA-Earth upstream template and adapted by Timon Geiss for the study configuration. Planning-horizon columns were added. DRC-specific wind-capacity limits were removed after verification that wind is non-extendable in all published scenarios and therefore does not enter the corresponding CCL constraints. | PyPSA-Earth/PyPSA-Eur and Timon Geiss attribution; CC-BY-4.0. |

## OpenStreetMap-derived network information

The custom transmission-network tables combine manually reconstructed study
data with selected information originating from OpenStreetMap.

For the line tables, the OSM-derived subset includes the following recurring
source identifiers:

`1028776989`, `1064077743`, `194187707`, `286737300`,
`384190117`, `865689670`, `224147292`, `460419487`,
`626214770`, `778351007` and `979570584`.

The remaining `figure_*` and `planned_*` line identifiers represent
study-specific reconstruction or planning records.

For the bus tables, the coordinates of Gbadolite, Kindu, Kisangani and Mbandaka
match the corresponding OpenStreetMap substation coordinates at the precision
used in the published CSV files. The voltage levels assigned to these buses are
study assumptions rather than OSM-derived values.

Because the OSM-derived and study-specific information is combined within the
published bus and line tables, these files are conservatively distributed under
the Open Database License (ODbL-1.0) with attribution to © OpenStreetMap
contributors. The study-specific reconstruction and transformations are
additionally attributed to Timon Geiss.

See:
https://www.openstreetmap.org/copyright

## Solar-profile generation

The repository does not redistribute the modified 2025 solar-profile NetCDF.

Instead, `scripts/scale_solar_profile.py` reproduces the required study input
from the standard PyPSA-Earth solar renewable profile. Only `p_nom_max` is
scaled, using one uniform factor to obtain a total installable PV capacity of
250 MW. All remaining profile variables are retained from the standard
PyPSA-Earth output.

The underlying PyPSA-Earth renewable-profile workflow uses ERA5 through atlite
together with spatial land-availability inputs. The relevant third-party
datasets retain their original licences and attribution requirements. The
generated NetCDF therefore remains a local workflow product rather than a
redistributed research-data file.

ERA5 information:
https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels

## Inputs not redistributed

The following intermediate or rebuild artefacts are intentionally excluded from
the publication repository:

- `data/custom/drc_myopic/osm/custom_substations_2025.geojson`
- `data/custom/drc_myopic/osm/custom_substations_2035.geojson`
- `data/custom/drc_myopic/renewable_profiles/solar_custom_2025.nc`

The custom-substation GeoJSON files are not required by the published workflow,
which directly uses the year-specific base-network CSV files.

The custom solar NetCDF is generated locally through the documented Snakemake
target and is explicitly ignored by Git.

## Reproducibility note

The custom input files contained in this repository represent the processed
model inputs used for the study rather than authoritative infrastructure or
geospatial datasets. Their scientific basis, modelling assumptions and major
source datasets are documented here and in the accompanying manuscript and
Supplementary Material.

Large baseline datasets required by the standard PyPSA-Earth workflow are not
redistributed in this repository and must be obtained through the corresponding
PyPSA-Earth data-retrieval workflow or from their original providers.
