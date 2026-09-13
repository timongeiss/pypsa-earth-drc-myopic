# SPDX-FileCopyrightText: 2026 Timon Geiss
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Scale only ``p_nom_max`` in a PyPSA-Earth renewable profile dataset."""

import argparse
import math
from pathlib import Path

import xarray as xr


def scale_p_nom_max(input_path, output_path, target_mw):
    """Write a copy of *input_path* whose total ``p_nom_max`` is *target_mw*."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    target_mw = float(target_mw)

    if not math.isfinite(target_mw) or target_mw <= 0:
        raise ValueError("target_mw must be a positive finite number")

    with xr.open_dataset(input_path) as source:
        dataset = source.load()

    if "p_nom_max" not in dataset:
        raise KeyError(f"{input_path} does not contain p_nom_max")

    source_total_mw = float(dataset["p_nom_max"].sum(skipna=True).item())
    if not math.isfinite(source_total_mw) or source_total_mw <= 0:
        raise ValueError(
            f"p_nom_max in {input_path} has invalid total {source_total_mw!r}"
        )

    scaling_factor = target_mw / source_total_mw
    dataset["p_nom_max"] = dataset["p_nom_max"] * scaling_factor

    scaled_total_mw = float(dataset["p_nom_max"].sum(skipna=True).item())
    if not math.isclose(scaled_total_mw, target_mw, rel_tol=1e-12, abs_tol=1e-9):
        raise RuntimeError(
            f"scaled p_nom_max total is {scaled_total_mw}, expected {target_mw}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(output_path)
    print(
        f"Scaled p_nom_max from {source_total_mw:.12g} MW to "
        f"{scaled_total_mw:.12g} MW (factor {scaling_factor:.16g}); "
        f"wrote {output_path}."
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="standard PyPSA-Earth profile_solar.nc")
    parser.add_argument("output", help="path for the scaled NetCDF file")
    parser.add_argument("--target-mw", type=float, default=250.0)
    return parser.parse_args()


if __name__ == "__main__":
    if "snakemake" in globals():
        scale_p_nom_max(
            snakemake.input.source,
            snakemake.output.profile,
            snakemake.params.target_mw,
        )
    else:
        args = parse_args()
        scale_p_nom_max(args.input, args.output, args.target_mw)
