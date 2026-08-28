#!/usr/bin/env python3
"""
This script generates quick-look figures for the DRC myopic H2G run.
It reads solved myopic postnetworks after the standalone off-grid optimisation
has been merged back into the network.

"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import shutil
from typing import Any

# Keep Matplotlib cache files outside the repository.
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/matplotlib-codex")))

import geopandas as gpd
import matplotlib

# Use a non-interactive backend because this script only writes files.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Wedge
import pandas as pd
from PIL import Image
import pypsa
from shapely import wkt
import yaml


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "results" / "H2G_A_CD_myopic" / "figures"

# Figure export settings copied from the original notebook backend.
FIGURE_DPI = 120
EXPORT_DPI = 300
A4_SQUARE_FIGSIZE = (7.2, 7.2)
TWO_COLUMN_FIGSIZE = A4_SQUARE_FIGSIZE
TWO_COLUMN_IMAGE_SIZE_PX = (2130, 2100)
TWO_COLUMN_EXPORT_FIGSIZE = (
    TWO_COLUMN_IMAGE_SIZE_PX[0] / EXPORT_DPI,
    TWO_COLUMN_IMAGE_SIZE_PX[1] / EXPORT_DPI,
)
DEMAND_COVERAGE_IMAGE_SIZE_PX = (TWO_COLUMN_IMAGE_SIZE_PX[0], TWO_COLUMN_IMAGE_SIZE_PX[1] // 2)
DEMAND_COVERAGE_FIGSIZE = (
    DEMAND_COVERAGE_IMAGE_SIZE_PX[0] / EXPORT_DPI,
    DEMAND_COVERAGE_IMAGE_SIZE_PX[1] / EXPORT_DPI,
)
AXIS_FONT_SIZE = 8.5
LEGEND_FONT_SIZE = 11.5
NODE_MARKER_SIZE = 52
NODE_LABEL_SIZE = 11.5
RESULT_LABEL_SIZE = NODE_LABEL_SIZE
FLOW_LABEL_SIZE = NODE_LABEL_SIZE
LINE_WIDTH = 1.75
CONVERTER_LINE_WIDTH = 1.85
CONVERTER_MARKER_SIZE = 70
RESULT_BUS_MARKER_SIZE = 64
MIN_BRANCH_LINE_WIDTH = 0.7
MAX_BRANCH_LINE_WIDTH = 5.2
COUNTRY_OUTLINE_LINE_WIDTH = 1.1
FRAME_LINE_WIDTH = 1.0
RESULT_MAP_PAD_RATIO = 0.005
H2_PIPELINE_FLOW_DISPLAY_THRESHOLD_TWH = 0.01
H2_EXPORT_DISPLAY_THRESHOLD_TWH = 0.01
H2_LHV_KWH_PER_KG = 33.333

plt.rcParams["figure.dpi"] = FIGURE_DPI
plt.rcParams["savefig.dpi"] = EXPORT_DPI

YEAR_STYLES = {2025: "-", 2035: "--", 2050: ":"}
YEAR_COLORS = {2025: "#1f4e79", 2035: "#9c6644", 2050: "#6a994e"}
DC_TOPOLOGY_COLOR = "#d12248"
CONVERTER_COLOR = "#264653"

# Bus marker colors for the access-region clusters.
ACCESS_REGION_COLORS = {
    "South-West": "#1b9e77",
    "East": "#d95f02",
    "North-Center": "#7570b3",
}

# One entry per myopic postnetwork that is loaded and plotted.
MODEL_SPECS = (
    {
        "key": "myopic_2025_0",
        "year": 2025,
        "label": "2025",
        "scenario_name": "H2G_A_CD_myopic",
        "network_path": (
            "results/H2G_A_CD_myopic/offgrid/postnetworks/"
            "elec_s_all_ec_lcopt_CCL_1H_2025_0.1046_NZ_0export_offgrid.nc"
        ),
    },
    {
        "key": "myopic_2035_0",
        "year": 2035,
        "label": "2035",
        "scenario_name": "H2G_A_CD_myopic",
        "network_path": (
            "results/H2G_A_CD_myopic/offgrid/postnetworks/"
            "elec_s_all_ec_lcopt_CCL_1H_2035_0.1046_NZ_0export_offgrid.nc"
        ),
    },
    {
        "key": "myopic_2035_early_large",
        "year": 2035,
        "label": "2035 early large",
        "scenario_name": "H2G_A_CD_myopic",
        "network_path": (
            "results/H2G_A_CD_myopic/offgrid/postnetworks/"
            "elec_s_all_ec_lcopt_CCL_1H_2035_0.1046_NZ_early_largeexport_offgrid.nc"
        ),
    },
    {
        "key": "myopic_2050_no_large_hydro",
        "year": 2050,
        "label": "2050 no large",
        "scenario_name": "H2G_A_CD_myopic",
        "network_path": (
            "results/H2G_A_CD_myopic/offgrid/postnetworks/"
            "elec_s_all_ec_lcopt_CCL_1H_2050_0.1046_NZ_no_large_hydroexport_offgrid.nc"
        ),
    },
    {
        "key": "myopic_2050_0",
        "year": 2050,
        "label": "2050 zero",
        "scenario_name": "H2G_A_CD_myopic",
        "network_path": (
            "results/H2G_A_CD_myopic/offgrid/postnetworks/"
            "elec_s_all_ec_lcopt_CCL_1H_2050_0.1046_NZ_0export_offgrid.nc"
        ),
    },
    {
        "key": "myopic_2050_23p33",
        "year": 2050,
        "label": "2050 low",
        "scenario_name": "H2G_A_CD_myopic",
        "network_path": (
            "results/H2G_A_CD_myopic/offgrid/postnetworks/"
            "elec_s_all_ec_lcopt_CCL_1H_2050_0.1046_NZ_23.33export_offgrid.nc"
        ),
    },
    {
        "key": "myopic_2050_78p33",
        "year": 2050,
        "label": "2050 mid",
        "scenario_name": "H2G_A_CD_myopic",
        "network_path": (
            "results/H2G_A_CD_myopic/offgrid/postnetworks/"
            "elec_s_all_ec_lcopt_CCL_1H_2050_0.1046_NZ_78.33export_offgrid.nc"
        ),
    },
    {
        "key": "myopic_2050_133p32",
        "year": 2050,
        "label": "2050 high",
        "scenario_name": "H2G_A_CD_myopic",
        "network_path": (
            "results/H2G_A_CD_myopic/offgrid/postnetworks/"
            "elec_s_all_ec_lcopt_CCL_1H_2050_0.1046_NZ_133.32export_offgrid.nc"
        ),
    },
)

# Short station names used in all maps.
DISPLAY_LABELS = {
    "BANDUNDU": "BANDUNDU",
    "BOMA": "BOMA",
    "BUKAVU_RUZIZI": "BUKAVU",
    "BUMBA": "BUMBA",
    "GBADOLITE": "GBADOLITE",
    "GOMA": "GOMA",
    "ILEBO": "ILEBO",
    "ISIRO": "ISIRO",
    "KALEMIE": "KALEMIE",
    "KAMINA": "KAMINA",
    "KANANGA": "KANANGA",
    "KIKWIT_KAKOBOLA": "KAKOBOLA",
    "KINDU": "KINDU",
    "KINSHASA": "KINSHASA",
    "KISANGANI": "KISANGANI",
    "KISENGE": "KISENGE",
    "KOLWEZI": "KOLWEZI",
    "LIKASI": "LIKASI",
    "LUBUMBASHI": "LUBUMBASHI",
    "MANONO": "MANONO",
    "MATADI_INGA": "MATADI-INGA",
    "MBANDAKA": "MBANDAKA",
    "MBUJI_MAYI": "MBUJI-MAYI",
    "SEMLIKI_IVUGHA": "SEMLIKI",
    "TSHIKAPA": "TSHIKAPA",
    "ZONGO": "ZONGO",
}

# Manual label offsets keep station names readable on the map.
LABEL_STYLES = {
    "BANDUNDU": {"dx": -0.10, "dy": 0.10, "ha": "right", "va": "bottom"},
    "BOMA": {"dx": -0.40, "dy": -0.28, "ha": "left", "va": "top"},
    "BUKAVU_RUZIZI": {"dx": 0.08, "dy": -0.02, "ha": "left", "va": "center"},
    "BUMBA": {"dx": 0.10, "dy": 0.08, "ha": "left", "va": "bottom"},
    "GBADOLITE": {"dx": 0.10, "dy": 0.08, "ha": "left", "va": "bottom"},
    "GOMA": {"dx": 0.08, "dy": 0.02, "ha": "left", "va": "bottom"},
    "ILEBO": {"dx": 0.10, "dy": 0.08, "ha": "left", "va": "bottom"},
    "ISIRO": {"dx": 0.10, "dy": 0.08, "ha": "left", "va": "bottom"},
    "KALEMIE": {"dx": 0.10, "dy": 0.08, "ha": "left", "va": "bottom"},
    "KAMINA": {"dx": 0.10, "dy": 0.08, "ha": "left", "va": "bottom"},
    "KANANGA": {"dx": -0.08, "dy": -0.12, "ha": "center", "va": "top"},
    "KIKWIT_KAKOBOLA": {"dx": -0.12, "dy": -0.10, "ha": "right", "va": "top"},
    "KINDU": {"dx": 0.10, "dy": 0.08, "ha": "left", "va": "bottom"},
    "KINSHASA": {"dx": -0.12, "dy": 0.14, "ha": "right", "va": "bottom"},
    "KISANGANI": {"dx": 0.10, "dy": 0.08, "ha": "left", "va": "bottom"},
    "KISENGE": {"dx": 0.00, "dy": -0.10, "ha": "center", "va": "top"},
    "KOLWEZI": {"dx": 0.10, "dy": 0.08, "ha": "left", "va": "bottom"},
    "LIKASI": {"dx": 0.08, "dy": -0.08, "ha": "left", "va": "top"},
    "LUBUMBASHI": {"dx": 0.08, "dy": -0.08, "ha": "left", "va": "top"},
    "MANONO": {"dx": 0.10, "dy": 0.08, "ha": "left", "va": "bottom"},
    "MATADI_INGA": {"dx": 0.14, "dy": -0.18, "ha": "left", "va": "top"},
    "MBANDAKA": {"dx": 0.10, "dy": 0.08, "ha": "left", "va": "bottom"},
    "MBUJI_MAYI": {"dx": 0.14, "dy": 0.02, "ha": "left", "va": "center"},
    "SEMLIKI_IVUGHA": {"dx": 0.08, "dy": 0.08, "ha": "left", "va": "bottom"},
    "TSHIKAPA": {"dx": -0.12, "dy": -0.10, "ha": "right", "va": "top"},
    "ZONGO": {"dx": -0.35, "dy": 0.02, "ha": "right", "va": "center"},
}


# ---------------------------------------------------------------------------
# Data model and model loading
# ---------------------------------------------------------------------------

@dataclass
class Model:
    """Store one loaded PyPSA model and its scenario metadata.

    Parameters
    ----------
    key : str
        Unique identifier used for filenames and model selection.
    year : int
        Scenario year represented by the model.
    label : str
        Short human-readable scenario label.
    scenario_name : str
        Name of the PyPSA-Earth scenario folder.
    network_path : Path
        Absolute path to the NetCDF network file.
    network : Any
        Loaded ``pypsa.Network`` object.

    Attributes
    ----------
    key : str
        Unique identifier used for filenames and model selection.
    year : int
        Scenario year represented by the model.
    label : str
        Short human-readable scenario label.
    scenario_name : str
        Name of the PyPSA-Earth scenario folder.
    network_path : Path
        Absolute path to the NetCDF network file.
    network : Any
        Loaded ``pypsa.Network`` object.
    """

    key: str
    year: int
    label: str
    scenario_name: str
    network_path: Path
    network: Any


def load_models() -> list[Model]:
    """Load all myopic result models used in the result plots.

    Parameters
    ----------
    None

    Returns
    -------
    list[Model]
        Loaded model objects in the order defined by ``MODEL_SPECS``.
    """
    missing_paths = [
        PROJECT_ROOT / spec["network_path"]
        for spec in MODEL_SPECS
        if not (PROJECT_ROOT / spec["network_path"]).exists()
    ]
    if missing_paths:
        missing_list = "\n".join(f"- {path}" for path in missing_paths)
        raise FileNotFoundError(
            "Missing myopic off-grid result network(s). Run the off-grid "
            "postprocessing after the solved myopic postnetworks exist, then "
            "rerun this plotting script.\n\n"
            f"{missing_list}\n\n"
            "For the current DRC myopic setup, use:\n"
            "python scripts/build_offgrid_myopic.py"
        )

    models = []
    for spec in MODEL_SPECS:
        # Convert the relative path from the spec into an absolute project path.
        network_path = PROJECT_ROOT / spec["network_path"]

        # Load the NetCDF file now so later plot functions can reuse the model.
        models.append(
            Model(
                key=spec["key"],
                year=spec["year"],
                label=spec["label"],
                scenario_name=spec["scenario_name"],
                network_path=network_path,
                network=pypsa.Network(network_path),
            )
        )
    return models


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def demand_regions_path(year: int) -> Path:
    """Return the demand-region mapping for a myopic horizon."""
    candidates = (
        PROJECT_ROOT / "data" / f"demand_regions_{year}.csv",
        PROJECT_ROOT / "data" / "custom" / "drc_myopic" / f"demand_regions_{year}.csv",
    )
    for path in candidates:
        if path.exists():
            return path

    searched = "\n".join(f"- {path}" for path in candidates)
    raise FileNotFoundError(
        f"Missing demand-region mapping for {year}. Searched:\n{searched}"
    )


def _save_figure(
    fig: plt.Figure,
    filename: str,
    fill_canvas: bool = False,
    target_size_px: tuple[int, int] = TWO_COLUMN_IMAGE_SIZE_PX,
) -> Path:
    """Save one figure in the Elsevier result folder.

    Parameters
    ----------
    fig : plt.Figure
        Matplotlib figure object to save.
    filename : str
        Output filename inside ``results/elsevier``.
    fill_canvas : bool
        If true, save without Matplotlib's tight bounding box so the axis frame
        can sit directly at the PNG edge.
    target_size_px : tuple[int, int]
        Final PNG size in pixels.

    Returns
    -------
    Path
        Absolute path to the written PNG file.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / filename

    # Use a white background to avoid transparent figure backgrounds in papers.
    if fill_canvas:
        fig.savefig(output_path, dpi=EXPORT_DPI, facecolor="white", pad_inches=0)
    else:
        fig.savefig(output_path, dpi=EXPORT_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Keep the PNG natural size fixed for reproducible LaTeX exports.
    image = Image.open(output_path).convert("RGBA")
    if image.size != target_size_px:
        target_width, target_height = target_size_px
        fixed_image = Image.new("RGBA", target_size_px, (255, 255, 255, 255))
        crop_left = max((image.width - target_width) // 2, 0)
        crop_top = max((image.height - target_height) // 2, 0)
        cropped = image.crop(
            (
                crop_left,
                crop_top,
                crop_left + min(image.width, target_width),
                crop_top + min(image.height, target_height),
            )
        )
        paste_left = max((target_width - cropped.width) // 2, 0)
        paste_top = max((target_height - cropped.height) // 2, 0)
        fixed_image.paste(cropped, (paste_left, paste_top))
        fixed_image.save(output_path, dpi=(EXPORT_DPI, EXPORT_DPI))
    return output_path


def _draw_station_labels(
    ax: Any,
    labels: pd.DataFrame,
    x_column: str,
    y_column: str,
    color: str,
    font_size: float,
) -> None:
    """Draw station labels with the shared Elsevier label style.

    Parameters
    ----------
    ax : Any
        Matplotlib axis receiving the label text.
    labels : pd.DataFrame
        Table with ``station_id`` and coordinate columns.
    x_column : str
        Name of the x-coordinate column in ``labels``.
    y_column : str
        Name of the y-coordinate column in ``labels``.
    color : str
        Text color.
    font_size : float
        Label font size in points.

    Returns
    -------
    None
    """
    for row in labels.itertuples(index=False):
        # Apply the manually tuned offset for this station.
        style = LABEL_STYLES[row.station_id]
        ax.text(
            getattr(row, x_column) + style["dx"],
            getattr(row, y_column) + style["dy"],
            DISPLAY_LABELS[row.station_id],
            fontsize=font_size,
            color=color,
            ha=style["ha"],
            va=style["va"],
            zorder=5,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.2},
        )





# -----------------------------------------------------------
# Network Data
# -----------------------------------------------------------


def plot_network_topology_overlay(models: list[Model]) -> Path:
    """Plot the three-year network topology overlay and save it as PNG.

    Parameters
    ----------
    models : list[Model]
        Loaded offgrid models. The function keeps one model per year for the
        topology overlay.

    Returns
    -------
    Path
        Absolute path to ``network_topology_overlay.png``.
    """
    # Keep one representative model per year; the last 2050 model is used.
    selected_by_year = {}
    for model in models:
        selected_by_year[model.year] = model
    topology_models = [selected_by_year[year] for year in sorted(selected_by_year)]

    # Use the first scenario's country outline as the map background.
    country_path = (
        PROJECT_ROOT
        / "resources"
        / topology_models[0].scenario_name
        / "shapes"
        / "country_shapes.geojson"
    )
    country = gpd.read_file(country_path).to_crs(epsg=4326)

    # Create the map canvas and set bounds around the DRC outline.
    fig, ax = plt.subplots(figsize=TWO_COLUMN_FIGSIZE, dpi=FIGURE_DPI)
    country.boundary.plot(
        ax=ax,
        color="#444444",
        linewidth=COUNTRY_OUTLINE_LINE_WIDTH,
        zorder=1,
    )
    xmin, ymin, xmax, ymax = country.total_bounds
    ax.set_xlim(xmin - max((xmax - xmin) * 0.05, 0.1), xmax + max((xmax - xmin) * 0.05, 0.1))
    ax.set_ylim(ymin - max((ymax - ymin) * 0.05, 0.1), ymax + max((ymax - ymin) * 0.05, 0.1))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude", fontsize=AXIS_FONT_SIZE)
    ax.set_ylabel("Latitude", fontsize=AXIS_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=AXIS_FONT_SIZE)

    # Start the legend with one node marker entry per displayed year.
    handles = []
    for model in topology_models:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=YEAR_COLORS[model.year],
                markeredgecolor="white",
                markeredgewidth=1.0,
                markersize=7.5,
                label=f"{model.year} nodes",
            )
        )

    previous_buses = None
    previous_lines = None
    previous_converters = None

    for model in topology_models:
        # Base-network CSV files contain the planned topology for each year.
        base_dir = PROJECT_ROOT / "resources" / model.scenario_name / "base_network"
        buses = pd.read_csv(base_dir / "custom_all_buses_build_network.csv")
        lines = pd.read_csv(base_dir / "custom_all_lines_build_network.csv")
        converters = pd.read_csv(base_dir / "custom_all_converters_build_network.csv")

        # The topology CSVs store line paths as WKT strings.
        for frame in (lines, converters):
            parsed_segments = []
            for geometry_text in frame["geometry"]:
                geometry = wkt.loads(str(geometry_text))
                if geometry.geom_type == "LineString":
                    parsed_segments.append([list(geometry.coords)])
                elif geometry.geom_type == "MultiLineString":
                    parsed_segments.append([list(line.coords) for line in geometry.geoms])
                else:
                    raise ValueError(f"Unsupported topology geometry: {geometry.geom_type}")
            frame["segments"] = parsed_segments

        # Plot only assets added since the previous year.
        if previous_buses is None:
            plot_buses = buses.copy()
            plot_lines = lines.copy()
            plot_converters = converters.copy()
        else:
            previous_bus_ids = set(previous_buses["bus_id"].astype(str))
            previous_line_ids = set(previous_lines["line_id"].astype(str))
            previous_converter_ids = set(previous_converters["converter_id"].astype(str))
            plot_buses = buses[~buses["bus_id"].astype(str).isin(previous_bus_ids)].copy()
            plot_lines = lines[~lines["line_id"].astype(str).isin(previous_line_ids)].copy()
            plot_converters = converters[
                ~converters["converter_id"].astype(str).isin(previous_converter_ids)
            ].copy()

        year = model.year
        linestyle = YEAR_STYLES[year]
        ac_color = YEAR_COLORS[year]
        line_mask = plot_lines["dc"].astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
        ac_lines = plot_lines[~line_mask].copy()
        dc_lines = plot_lines[line_mask].copy()

        # Draw AC and DC line segments with the current year's style.
        for rows, color, label in (
            (ac_lines, ac_color, f"{year} line"),
            (dc_lines, DC_TOPOLOGY_COLOR, f"{year} DC line"),
        ):
            if rows.empty:
                continue
            for row in rows.itertuples(index=False):
                for segment in row.segments:
                    xs = [point[0] for point in segment]
                    ys = [point[1] for point in segment]
                    ax.plot(
                        xs,
                        ys,
                        color=color,
                        linestyle=linestyle,
                        linewidth=LINE_WIDTH,
                        alpha=0.95,
                        zorder=2,
                    )
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=color,
                    linestyle=linestyle,
                    linewidth=LINE_WIDTH,
                    label=label,
                )
            )

        # Draw converter paths and mark their midpoints with an x symbol.
        if not plot_converters.empty:
            for row in plot_converters.itertuples(index=False):
                for segment in row.segments:
                    xs = [point[0] for point in segment]
                    ys = [point[1] for point in segment]
                    ax.plot(
                        xs,
                        ys,
                        color=CONVERTER_COLOR,
                        linestyle=linestyle,
                        linewidth=CONVERTER_LINE_WIDTH,
                        alpha=0.95,
                        zorder=3,
                    )
                midpoint = row.segments[0][len(row.segments[0]) // 2]
                ax.scatter(
                    [midpoint[0]],
                    [midpoint[1]],
                    marker="x",
                    s=CONVERTER_MARKER_SIZE,
                    color=CONVERTER_COLOR,
                    linewidths=1.5,
                    zorder=5,
                )
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=CONVERTER_COLOR,
                    linestyle="-",
                    marker="x",
                    markersize=7.5,
                    linewidth=CONVERTER_LINE_WIDTH,
                    label="Converters",
                )
            )

        # Draw new buses and label them with the compact station names.
        if not plot_buses.empty:
            ax.scatter(
                plot_buses["lon"],
                plot_buses["lat"],
                s=NODE_MARKER_SIZE,
                marker="o",
                facecolors=ac_color,
                edgecolors="white",
                linewidths=1.3,
                zorder=4,
            )
            labels = plot_buses[["station_id", "lon", "lat"]].drop_duplicates()
            _draw_station_labels(ax, labels, "lon", "lat", ac_color, NODE_LABEL_SIZE)

        previous_buses = buses
        previous_lines = lines
        previous_converters = converters

    # Finish the plot.
    ax.legend(
        handles=handles,
        loc="lower left",
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        handlelength=1.5,
        columnspacing=0.9,
        handletextpad=0.4,
    )
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.tight_layout()
    return _save_figure(fig, "network_topology_overlay.png")















# -----------------------------------------------------------
# PyPSA Networks
# -----------------------------------------------------------

def plot_result_network_topology(
    model: Model,
    filename: str | None = None,
) -> Path:
    """Plot one result network with annual branch flows in TWh.

    Parameters
    ----------
    model : Model
        Loaded model to plot.
    filename : str | None
        Optional output filename. If omitted, the model key is added to the
        default result-network filename.

    Returns
    -------
    Path
        Absolute path to the written result-network PNG file.
    """
    network = model.network
    result_station_label_size = RESULT_LABEL_SIZE
    result_flow_label_size = FLOW_LABEL_SIZE
    result_bus_marker_size = RESULT_BUS_MARKER_SIZE * 0.5

    # The result-network plot only uses the country outline, not region shapes.
    resource_dir = PROJECT_ROOT / "resources" / model.scenario_name
    country = gpd.read_file(resource_dir / "shapes" / "country_shapes.geojson").to_crs(epsg=4326)
    xmin, ymin, xmax, ymax = country.total_bounds
    x_pad = max((xmax - xmin) * RESULT_MAP_PAD_RATIO, 0.02)
    y_pad = max((ymax - ymin) * RESULT_MAP_PAD_RATIO, 0.02)
    plot_xlim = (xmin - x_pad, xmax + x_pad)
    plot_ylim = (ymin - y_pad, ymax + y_pad)

    # Keep only geographic AC and DC buses.
    buses = network.buses.copy()
    buses = buses[buses["carrier"].astype(str).str.upper().isin(["AC", "DC"])].copy()
    buses["x"] = pd.to_numeric(buses["x"], errors="coerce")
    buses["y"] = pd.to_numeric(buses["y"], errors="coerce")
    buses = buses.dropna(subset=["x", "y"])
    buses_ac = buses[buses["carrier"].astype(str).str.upper() == "AC"].copy()
    buses_dc = buses[buses["carrier"].astype(str).str.upper() == "DC"].copy()

    # Map buses to access-region clusters for marker colors.
    demand_regions = pd.read_csv(demand_regions_path(model.year))
    region_column = "access_region" if "access_region" in demand_regions.columns else "region"
    access_region_by_bus = demand_regions.drop_duplicates("bus").set_index("bus")[region_column]
    buses_ac["access_region"] = buses_ac.index.to_series().map(access_region_by_bus)
    buses_dc["access_region"] = buses_dc.index.to_series().map(access_region_by_bus)
    missing_dc_region = buses_dc["access_region"].isna()
    buses_dc.loc[missing_dc_region, "access_region"] = (
        buses_dc.loc[missing_dc_region]
        .index.astype(str)
        .str.replace("_DC_", "_AC_", regex=False)
        .map(access_region_by_bus)
    )

    # Use carrier colors from the solved network for fallback bus-marker colors.
    carrier_colors = network.carriers["color"].astype(str)
    ac_color = carrier_colors.at["AC"]
    dc_color = carrier_colors.at["DC"]
    lines = network.lines[network.lines["carrier"].astype(str).str.upper() == "AC"].copy()
    links = network.links[network.links["carrier"].astype(str).str.upper() == "DC"].copy()

    # Convert weighted hourly flows from MWh to annual TWh per branch.
    weights = network.snapshot_weightings["objective"]
    line_energy = (
        network.lines_t.p0.reindex(columns=lines.index, fill_value=0.0)
        .abs()
        .multiply(weights, axis=0)
        .sum()
        / 1e6
    )
    link_energy = (
        network.links_t.p0.reindex(columns=links.index, fill_value=0.0)
        .abs()
        .multiply(weights, axis=0)
        .sum()
        / 1e6
    )
    max_energy_twh = float(pd.concat([line_energy, link_energy]).max())
    max_energy_label = str(math.ceil(max(max_energy_twh, 6.0)))
    branch_flow_classes = [
        ("<1 TWh/a", 0.0, 1.0, "#218c3e"),
        ("1-3 TWh/a", 1.0, 3.0, "#f5b82e"),
        ("3-6 TWh/a", 3.0, 6.0, "#e86e1c"),
        (f"6-{max_energy_label} TWh/a", 6.0, float("inf"), "#c71f1f"),
    ]

    def branch_color_from_twh(value: float) -> str:
        """Map annual branch energy to a fixed categorical color.

        Parameters
        ----------
        value : float
            Annual branch flow in TWh.

        Returns
        -------
        str
            Hex color for Matplotlib.
        """
        for _, lower, upper, color in branch_flow_classes:
            if lower <= float(value) < upper:
                return color
        return branch_flow_classes[-1][3]

    def station_id_from_bus_name(bus_name: Any) -> str:
        """Convert an AC/DC bus name to a station id.

        Parameters
        ----------
        bus_name : Any
            PyPSA bus index value.

        Returns
        -------
        str
            Station id without carrier and year suffix.
        """
        station_id = str(bus_name)
        for carrier in ("AC", "DC"):
            for year in (2025, 2035, 2050):
                suffix = f"_{carrier}_{year}"
                if station_id.endswith(suffix):
                    station_id = station_id[: -len(suffix)]
        return station_id

    def draw_branch(
        row: pd.Series,
        energy_twh: float,
        zorder: int,
        line_style: str | tuple[int, tuple[int, int]],
        label_flow: bool,
    ) -> None:
        """Draw one branch using annual energy as color intensity.

        Parameters
        ----------
        row : pd.Series
            Row from ``network.lines`` or ``network.links``.
        energy_twh : float
            Annual branch flow in TWh.
        zorder : int
            Matplotlib drawing order.
        line_style : str | tuple[int, tuple[int, int]]
            Matplotlib line style. AC is solid, DC is dashed.
        label_flow : bool
            Whether to write the branch flow next to this branch.

        Returns
        -------
        None
        """
        if row.bus0 not in buses.index or row.bus1 not in buses.index:
            return
        bus0 = buses.loc[row.bus0]
        bus1 = buses.loc[row.bus1]

        # Draw the branch between the two bus coordinates.
        ax.plot(
            [bus0.x, bus1.x],
            [bus0.y, bus1.y],
            color=branch_color_from_twh(energy_twh),
            linewidth=1.5,
            linestyle=line_style,
            alpha=0.95,
            solid_capstyle="round",
            dash_capstyle="round",
            zorder=zorder,
        )

        if label_flow:
            label_x = (bus0.x + bus1.x) / 2
            label_y = (bus0.y + bus1.y) / 2
            label_ha = "center"
            if label_x < plot_xlim[0] + 1.0:
                label_x = plot_xlim[0] + 0.18
                label_ha = "left"
            elif label_x > plot_xlim[1] - 1.0:
                label_x = plot_xlim[1] - 0.18
                label_ha = "right"
            ax.text(
                label_x,
                label_y,
                f"{energy_twh:.1f} TWh",
                fontsize=result_flow_label_size,
                color="#1F2933",
                ha=label_ha,
                va="center",
                zorder=8,
            )

    # Create a clean map with only the country outline in the background.
    fig, ax = plt.subplots(figsize=TWO_COLUMN_EXPORT_FIGSIZE, dpi=FIGURE_DPI)
    ax.set_position([0.0, 0.0, 1.0, 1.0])
    ax.set_facecolor("white")
    country.boundary.plot(
        ax=ax,
        color="black",
        linewidth=COUNTRY_OUTLINE_LINE_WIDTH,
        zorder=1,
    )

    # Collect drawable branches first so they can be plotted consistently.
    branch_rows = []
    for branch_name, row in lines.iterrows():
        if row.bus0 in buses.index and row.bus1 in buses.index:
            branch_rows.append((row, float(line_energy.get(branch_name, 0.0)), 3, "-"))

    for branch_name, row in links.iterrows():
        if row.bus0 in buses.index and row.bus1 in buses.index:
            branch_rows.append((row, float(link_energy.get(branch_name, 0.0)), 4, (0, (4, 2))))

    # Draw branches from bus coordinates, as in the notebook's AC/DC topology plot.
    for row, energy_twh, zorder, line_style in branch_rows:
        draw_branch(
            row,
            energy_twh,
            zorder,
            line_style,
            label_flow=False,
        )

    # Draw AC/DC bus markers using access-region cluster colors.
    for region_name, color in ACCESS_REGION_COLORS.items():
        region_ac = buses_ac[buses_ac["access_region"] == region_name]
        region_dc = buses_dc[buses_dc["access_region"] == region_name]
        if not region_ac.empty:
            ax.scatter(
                region_ac["x"],
                region_ac["y"],
                s=result_bus_marker_size,
                marker="o",
                color=color,
                zorder=6,
            )
        if not region_dc.empty:
            ax.scatter(
                region_dc["x"],
                region_dc["y"],
                s=result_bus_marker_size,
                marker="s",
                color=color,
                zorder=6,
            )

    buses_ac_missing = buses_ac[buses_ac["access_region"].isna()]
    buses_dc_missing = buses_dc[buses_dc["access_region"].isna()]
    if not buses_ac_missing.empty:
        ax.scatter(
            buses_ac_missing["x"],
            buses_ac_missing["y"],
            s=result_bus_marker_size,
            marker="o",
            color=ac_color,
            zorder=6,
        )
    if not buses_dc_missing.empty:
        ax.scatter(
            buses_dc_missing["x"],
            buses_dc_missing["y"],
            s=result_bus_marker_size,
            marker="s",
            color=dc_color,
            zorder=6,
        )

    # Build labels from AC buses, then add DC-only stations if no AC bus exists.
    labels = buses_ac[["x", "y"]].copy()
    labels["station_id"] = [station_id_from_bus_name(bus_name) for bus_name in labels.index]
    ac_station_ids = set(labels["station_id"])
    dc_labels = buses_dc[["x", "y"]].copy()
    dc_labels["station_id"] = [station_id_from_bus_name(bus_name) for bus_name in dc_labels.index]
    dc_labels = dc_labels[~dc_labels["station_id"].isin(ac_station_ids)]
    labels = pd.concat([labels, dc_labels], axis=0)
    labels = labels[["station_id", "x", "y"]].drop_duplicates("station_id")
    _draw_station_labels(ax, labels, "x", "y", "#1F2933", result_station_label_size)

    flow_handles = [
        Line2D(
            [0],
            [0],
            color=color,
            lw=2.4,
            linestyle="-",
            label=label,
        )
        for label, _, _, color in branch_flow_classes
    ]
    flow_legend = ax.legend(
        handles=flow_handles,
        title="Annual branch flow",
        loc="upper left",
        frameon=True,
        fontsize=LEGEND_FONT_SIZE,
        title_fontsize=LEGEND_FONT_SIZE,
        handlelength=1.7,
        handletextpad=0.55,
        borderpad=0.55,
    )
    ax.add_artist(flow_legend)

    # Legend explains branch type, marker type, and access region.
    legend_branch_color = "#111827"
    handles = [
        Line2D([0], [0], color=legend_branch_color, lw=2.4, linestyle="-", label="AC"),
        Line2D(
            [0],
            [0],
            color=legend_branch_color,
            lw=2.4,
            linestyle=(0, (4, 2)),
            label="DC",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=9,
            label="AC bus marker",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=9,
            label="DC bus marker",
        ),
    ]
    for region_name, color in ACCESS_REGION_COLORS.items():
        has_region = (
            (not buses_ac.empty and (buses_ac["access_region"] == region_name).any())
            or (not buses_dc.empty and (buses_dc["access_region"] == region_name).any())
        )
        if has_region:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=color,
                    markersize=9,
                    label=region_name,
                )
            )
    # Use fixed country bounds and keep only the outer frame visible.
    ax.set_xlim(plot_xlim)
    ax.set_ylim(plot_ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(FRAME_LINE_WIDTH)
    ax.set_title("")
    ax.legend(
        handles=handles,
        loc="lower left",
        frameon=True,
        fontsize=LEGEND_FONT_SIZE,
        handlelength=1.5,
        columnspacing=0.9,
        handletextpad=0.4,
    )

    # Default filenames keep one file per model.
    if filename is None:
        filename = f"result_network_ac_dc_bus_topology_{model.key}.png"
    return _save_figure(fig, filename, fill_canvas=True)


def plot_h2_pipeline_flow_maps(models: list[Model]) -> list[Path]:
    """Plot H2 pipeline flow maps for the non-zero 2050 export scenarios.

    Parameters
    ----------
    models : list[Model]
        Loaded model objects in the order defined by ``MODEL_SPECS``.

    Returns
    -------
    list[Path]
        Absolute paths to the written H2 pipeline flow PNG files.
    """
    selected_keys = (
        "myopic_2050_23p33",
        "myopic_2050_78p33",
        "myopic_2050_133p32",
    )
    models_by_key = {model.key: model for model in models}
    selected_models = [
        models_by_key[key]
        for key in selected_keys
        if key in models_by_key
    ]
    if not selected_models:
        return []

    def h2_station_id_from_bus_name(bus_name: Any) -> str:
        """Convert an H2 bus name to the station id used by map labels."""
        station_id = str(bus_name).removesuffix(" H2")
        for carrier in ("AC", "DC"):
            for year in (2025, 2035, 2050):
                suffix = f"_{carrier}_{year}"
                if station_id.endswith(suffix):
                    station_id = station_id[: -len(suffix)]
        return station_id

    def h2_bus_table(network: Any) -> pd.DataFrame:
        """Return geographic H2 buses, excluding the synthetic export bus."""
        buses = network.buses.copy()
        carriers = buses.get("carrier", pd.Series("", index=buses.index)).astype(str)
        buses = buses[carriers.str.lower() == "h2"].copy()
        buses = buses[buses.index.astype(str).str.lower() != "h2 export bus"]
        buses["x"] = pd.to_numeric(buses["x"], errors="coerce")
        buses["y"] = pd.to_numeric(buses["y"], errors="coerce")
        return buses.dropna(subset=["x", "y"])

    def export_energy_by_node_twh(network: Any) -> pd.Series:
        """Return annual H2 export energy by source H2 bus in TWh."""
        links = network.links.copy()
        carriers = links.get("carrier", pd.Series("", index=links.index)).astype(str)
        export_links = links[
            carriers.str.lower().eq("h2")
            & links["bus1"].astype(str).eq("H2 export bus")
        ]
        if export_links.empty:
            return pd.Series(dtype=float)
        weights = network.snapshot_weightings["objective"]
        dispatch = network.links_t.p0.reindex(columns=export_links.index, fill_value=0.0)
        energy = dispatch.clip(lower=0.0).multiply(weights, axis=0).sum() / 1e6
        return energy.groupby(export_links["bus0"]).sum()

    def h2_pipeline_table(network: Any) -> pd.DataFrame:
        """Return built H2 pipeline links with annual flow values."""
        links = network.links.copy()
        carriers = links.get("carrier", pd.Series("", index=links.index)).astype(str)
        pipelines = links[carriers.str.lower().eq("h2 pipeline")].copy()
        if pipelines.empty:
            return pipelines

        capacity = pd.to_numeric(pipelines.get("p_nom_opt", 0.0), errors="coerce")
        if "p_nom" in pipelines.columns:
            fallback_capacity = pd.to_numeric(pipelines["p_nom"], errors="coerce")
            capacity = capacity.where(capacity > 0.0, fallback_capacity)
        pipelines["_capacity_mw"] = capacity.fillna(0.0)
        pipelines = pipelines[pipelines["_capacity_mw"] > 0.0].copy()
        if pipelines.empty:
            return pipelines

        weights = network.snapshot_weightings["objective"]
        annual_flow_twh = (
            network.links_t.p0.reindex(columns=pipelines.index, fill_value=0.0)
            .abs()
            .multiply(weights, axis=0)
            .sum()
            / 1e6
        )
        pipelines["_abs_flow_twh"] = annual_flow_twh.reindex(pipelines.index).fillna(0.0)
        pipelines = pipelines[
            pipelines["_abs_flow_twh"] >= H2_PIPELINE_FLOW_DISPLAY_THRESHOLD_TWH
        ].copy()
        pipelines["_abs_flow_gwh"] = pipelines["_abs_flow_twh"] * 1000.0
        return pipelines

    export_energy_by_model = {
        model.key: export_energy_by_node_twh(model.network)
        for model in selected_models
    }
    visible_export_values = []
    for export_energy in export_energy_by_model.values():
        visible_exports = export_energy[
            export_energy >= H2_EXPORT_DISPLAY_THRESHOLD_TWH
        ]
        if not visible_exports.empty:
            visible_export_values.append(visible_exports)
    if visible_export_values:
        global_export_energy = pd.concat(visible_export_values)
        global_min_export_twh = float(global_export_energy.min())
        global_max_export_twh = float(global_export_energy.max())
    else:
        global_min_export_twh = 0.0
        global_max_export_twh = 0.0

    def h2_pipeline_flow_classes(max_flow_twh: float) -> list[tuple[str, float, float, str]]:
        """Return compact H2 pipeline flow classes in annual TWh."""
        max_flow_label = str(math.ceil(max(max_flow_twh, 50.0)))
        return [
            ("<10 TWh/a", H2_PIPELINE_FLOW_DISPLAY_THRESHOLD_TWH, 10.0, "#218c3e"),
            ("10-50 TWh/a", 10.0, 50.0, "#e86e1c"),
            (f"50-{max_flow_label} TWh/a", 50.0, float("inf"), "#c71f1f"),
        ]

    def h2_pipeline_class_from_twh(
        value: float,
        flow_classes: list[tuple[str, float, float, str]],
    ) -> tuple[str, str]:
        """Map one annual H2 pipeline flow to a legend label and color."""
        for label, lower, upper, color in flow_classes:
            if lower <= float(value) < upper:
                return label, color
        return flow_classes[-1][0], flow_classes[-1][3]

    def format_twh(value: float) -> str:
        """Format annual TWh values compactly for map legends."""
        if value >= 10.0:
            return f"{value:.0f}"
        if value >= 1.0:
            return f"{value:.1f}"
        return f"{value:.2f}"

    pipeline_tables_by_model = {
        model.key: h2_pipeline_table(model.network)
        for model in selected_models
    }
    visible_pipeline_flow = [
        table["_abs_flow_twh"]
        for table in pipeline_tables_by_model.values()
        if not table.empty
    ]
    if visible_pipeline_flow:
        global_pipeline_flow = pd.concat(visible_pipeline_flow)
        global_max_flow_twh = float(global_pipeline_flow.max())
    else:
        global_pipeline_flow = pd.Series(dtype=float)
        global_max_flow_twh = 0.0
    pipeline_flow_classes = h2_pipeline_flow_classes(global_max_flow_twh)
    global_pipeline_class_labels = {
        h2_pipeline_class_from_twh(float(value), pipeline_flow_classes)[0]
        for value in global_pipeline_flow
    }

    def export_star_size(export_twh: float, min_export_twh: float, max_export_twh: float) -> float:
        """Scale export stars continuously across the selected scenarios."""
        min_size = 75.0
        max_size = 300.0
        if max_export_twh <= min_export_twh:
            return max_size
        fraction = (export_twh - min_export_twh) / (max_export_twh - min_export_twh)
        return min_size + fraction * (max_size - min_size)

    output_paths = []

    for model in selected_models:
        network = model.network
        resource_dir = PROJECT_ROOT / "resources" / model.scenario_name
        country = gpd.read_file(resource_dir / "shapes" / "country_shapes.geojson").to_crs(epsg=4326)
        xmin, ymin, xmax, ymax = country.total_bounds
        x_pad = max((xmax - xmin) * RESULT_MAP_PAD_RATIO, 0.02)
        y_pad = max((ymax - ymin) * RESULT_MAP_PAD_RATIO, 0.02)
        plot_xlim = (xmin - x_pad, xmax + x_pad)
        plot_ylim = (ymin - y_pad, ymax + y_pad)

        h2_buses = h2_bus_table(network)
        pipelines = pipeline_tables_by_model[model.key]
        export_energy = export_energy_by_model[model.key]
        export_nodes = export_energy[export_energy >= H2_EXPORT_DISPLAY_THRESHOLD_TWH]
        no_export_nodes = export_energy[export_energy < H2_EXPORT_DISPLAY_THRESHOLD_TWH]
        carrier_colors = network.carriers.get("color", pd.Series(dtype=str)).astype(str)
        h2_color = carrier_colors.get("H2", "#16a3c7")
        export_color = carrier_colors.get("H2 export", "#c026d3")

        fig, ax = plt.subplots(figsize=TWO_COLUMN_EXPORT_FIGSIZE, dpi=FIGURE_DPI)
        ax.set_position([0.0, 0.0, 1.0, 1.0])
        ax.set_facecolor("white")
        country.boundary.plot(
            ax=ax,
            color="black",
            linewidth=COUNTRY_OUTLINE_LINE_WIDTH,
            zorder=1,
        )

        for _, row in pipelines.iterrows():
            if row.bus0 not in h2_buses.index or row.bus1 not in h2_buses.index:
                continue
            bus0 = h2_buses.loc[row.bus0]
            bus1 = h2_buses.loc[row.bus1]
            flow_twh = float(row["_abs_flow_twh"])
            _, pipeline_line_color = h2_pipeline_class_from_twh(
                flow_twh,
                pipeline_flow_classes,
            )
            ax.plot(
                [bus0.x, bus1.x],
                [bus0.y, bus1.y],
                color=pipeline_line_color,
                linewidth=2.8,
                linestyle="-",
                alpha=0.95,
                solid_capstyle="round",
                zorder=6,
            )

        if not h2_buses.empty:
            ax.scatter(
                h2_buses["x"],
                h2_buses["y"],
                s=RESULT_BUS_MARKER_SIZE * 0.35,
                marker="o",
                color=h2_color,
                edgecolors="white",
                linewidths=0.8,
                zorder=5,
            )

        for bus_name, export_twh in no_export_nodes.items():
            if bus_name not in h2_buses.index:
                continue
            bus = h2_buses.loc[bus_name]
            ax.scatter(
                [bus.x],
                [bus.y],
                s=75.0,
                marker="*",
                facecolors="white",
                edgecolors=export_color,
                linewidths=1.0,
                zorder=7,
            )

        for bus_name, export_twh in export_nodes.items():
            if bus_name not in h2_buses.index:
                continue
            bus = h2_buses.loc[bus_name]
            marker_size = export_star_size(
                float(export_twh),
                global_min_export_twh,
                global_max_export_twh,
            )
            ax.scatter(
                [bus.x],
                [bus.y],
                s=marker_size,
                marker="*",
                color=export_color,
                edgecolors="black",
                linewidths=0.7,
                zorder=8,
            )

        labels = h2_buses[["x", "y"]].copy()
        labels["station_id"] = [h2_station_id_from_bus_name(bus_name) for bus_name in labels.index]
        labels = labels[labels["station_id"].isin(DISPLAY_LABELS)]
        labels = labels[["station_id", "x", "y"]].drop_duplicates("station_id")
        _draw_station_labels(ax, labels, "x", "y", "#1F2933", RESULT_LABEL_SIZE)

        flow_handles = [
            Line2D(
                [0],
                [0],
                color=color,
                lw=2.4,
                linestyle="-",
                label=label,
            )
            for label, _, _, color in pipeline_flow_classes
            if label in global_pipeline_class_labels
        ]
        if flow_handles:
            flow_legend = ax.legend(
                handles=flow_handles,
                title="Annual H2 flow",
                loc="upper left",
                frameon=True,
                fontsize=LEGEND_FONT_SIZE,
                title_fontsize=LEGEND_FONT_SIZE,
                handlelength=1.7,
                handletextpad=0.55,
                borderpad=0.55,
            )
            ax.add_artist(flow_legend)

        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=h2_color,
                markeredgecolor="white",
                markersize=7,
                label="H2 bus",
            ),
        ]
        if not no_export_nodes.empty:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="*",
                    color="w",
                    markerfacecolor="white",
                    markeredgecolor=export_color,
                    markersize=(75.0 ** 0.5) * 0.65,
                    label="No export (<0.01 TWh/a)",
                )
            )
        if global_max_export_twh > 0.0:
            export_legend_values = [(global_min_export_twh, "Export scale min")]
            if global_max_export_twh > global_min_export_twh:
                export_legend_values.append((global_max_export_twh, "Export scale max"))
            for export_twh, label_prefix in export_legend_values:
                marker_size = export_star_size(
                    export_twh,
                    global_min_export_twh,
                    global_max_export_twh,
                )
                handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker="*",
                        color="w",
                        markerfacecolor=export_color,
                        markeredgecolor="black",
                        markersize=(marker_size ** 0.5) * 0.65,
                        label=f"{label_prefix}: {format_twh(export_twh)} TWh/a",
                    )
                )

        ax.set_xlim(plot_xlim)
        ax.set_ylim(plot_ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(FRAME_LINE_WIDTH)
        ax.set_title("")
        ax.legend(
            handles=handles,
            loc="lower left",
            frameon=True,
            fontsize=LEGEND_FONT_SIZE,
            handlelength=1.5,
            columnspacing=0.9,
            handletextpad=0.4,
        )

        output_path = _save_figure(
            fig,
            f"h2_pipeline_flow_map_{model.key}.png",
            fill_canvas=True,
        )
        legacy_output_path = OUTPUT_DIR / f"h2_pipeline_utilization_map_{model.key}.png"
        shutil.copyfile(output_path, legacy_output_path)
        output_paths.append(output_path)

    return output_paths


def plot_h2_pipeline_utilization_maps(models: list[Model]) -> list[Path]:
    """Backward-compatible wrapper for the H2 pipeline flow maps."""
    return plot_h2_pipeline_flow_maps(models)


def plot_electricity_demand_node_maps(models: list[Model]) -> list[Path]:
    """Plot node-level industry, residual on-grid, and off-grid demand maps.

    Parameters
    ----------
    models : list[Model]
        Loaded model objects in the order defined by ``MODEL_SPECS``.

    Returns
    -------
    list[Path]
        Absolute paths to the written demand-map PNG files.
    """
    selected_keys = (
        "myopic_2025_0",
        "myopic_2035_0",
        "myopic_2035_early_large",
        "myopic_2050_no_large_hydro",
    )
    models_by_key = {model.key: model for model in models}
    selected_models = [models_by_key[key] for key in selected_keys if key in models_by_key]
    if not selected_models:
        return []

    industry_color = "#111827"
    residual_color = "#2a6f97"
    offgrid_color = "#e76f51"
    max_pie_radius = 0.55

    def profile_timestep_hours(index: pd.Index) -> float:
        """Infer the time step length of a CSV demand profile in hours."""
        timestamps = pd.to_datetime(index, errors="coerce")
        if pd.isna(timestamps).any():
            return 1.0
        diffs = pd.DatetimeIndex(timestamps).to_series().diff().dropna()
        if diffs.empty:
            return 1.0
        hours = diffs.median().total_seconds() / 3600.0
        return float(hours) if hours > 0.0 else 1.0

    def station_id_from_bus_name(bus_name: Any) -> str:
        """Convert an AC bus name to a station id."""
        station_id = str(bus_name)
        for carrier in ("AC", "DC"):
            for year in (2025, 2035, 2050):
                suffix = f"_{carrier}_{year}"
                if station_id.endswith(suffix):
                    station_id = station_id[: -len(suffix)]
        return station_id

    def gamma_industrial(model: Model) -> float:
        """Read the scenario's ex-ante industrial demand share."""
        config_path = PROJECT_ROOT / "configs" / "scenarios_H2G" / f"config.{model.scenario_name}.yaml"
        with open(config_path, encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        gamma = (
            config.get("load_options", {})
            .get("demand_allocation", {})
            .get("gamma_industrial", 0.30)
        )
        return float(gamma)

    def access_region_by_bus(model: Model) -> pd.Series:
        """Return the demand-access-region assignment by bus for one horizon."""
        try:
            path = demand_regions_path(model.year)
        except FileNotFoundError:
            return pd.Series(dtype=object)

        regions = pd.read_csv(path)
        if "bus" not in regions.columns:
            return pd.Series(dtype=object)
        region_column = "access_region" if "access_region" in regions.columns else "region"
        if region_column not in regions.columns:
            return pd.Series(dtype=object)

        regions = regions.dropna(subset=["bus", region_column]).drop_duplicates("bus")
        return regions.set_index("bus")[region_column].astype(str)

    def demand_by_bus_twh(model: Model) -> pd.DataFrame:
        """Return annual industry, residual on-grid, and off-grid demand by AC bus."""
        resource_dir = PROJECT_ROOT / "resources" / model.scenario_name
        rows = []
        for filename, column_name in (
            ("demand_profiles.csv", "on_grid"),
            ("offgrid_demand_profiles.csv", "off_grid"),
        ):
            candidates = [
                resource_dir / f"horizon_{model.year}" / filename,
                resource_dir / filename,
            ]
            path = next((candidate for candidate in candidates if candidate.exists()), None)
            if path is None:
                demand = pd.Series(dtype=float)
            else:
                profile = pd.read_csv(path, index_col=0).apply(pd.to_numeric, errors="raise")
                demand = profile.sum(axis=0) * profile_timestep_hours(profile.index) / 1e6
            rows.append(demand.rename(column_name))

        demand = pd.concat(rows, axis=1).fillna(0.0)
        demand["total"] = demand["on_grid"] + demand["off_grid"]
        demand = demand[demand["total"] > 0.0].copy()
        gamma = gamma_industrial(model)
        demand["industry"] = demand["total"] * gamma
        demand["residual_on_grid"] = (demand["on_grid"] - demand["industry"]).clip(lower=0.0)
        demand["access_region"] = demand.index.to_series().map(access_region_by_bus(model))

        buses = model.network.buses.copy()
        buses["x"] = pd.to_numeric(buses["x"], errors="coerce")
        buses["y"] = pd.to_numeric(buses["y"], errors="coerce")
        demand = demand.join(buses[["x", "y"]], how="left")
        return demand.dropna(subset=["x", "y"])

    def access_region_shapes(model: Model) -> gpd.GeoDataFrame:
        """Return demand-region polygons with an access-region assignment.

        The bus-region files only cover modelled demand regions. Remaining
        GADM polygons are assigned by their strongest overlap with the
        classified bus regions so the map background covers the full country.
        """
        region_assignment = access_region_by_bus(model)
        if region_assignment.empty:
            return gpd.GeoDataFrame(columns=["access_region", "geometry"], geometry="geometry", crs="EPSG:4326")

        resource_dir = PROJECT_ROOT / "resources" / model.scenario_name
        candidates = [
            resource_dir / f"horizon_{model.year}" / "bus_regions" / "regions_onshore_elec_s_all.geojson",
            resource_dir / f"horizon_{model.year}" / "bus_regions" / "regions_onshore.geojson",
        ]
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if path is None:
            return gpd.GeoDataFrame(columns=["access_region", "geometry"], geometry="geometry", crs="EPSG:4326")

        regions = gpd.read_file(path)
        if "name" not in regions.columns or regions.empty:
            return gpd.GeoDataFrame(columns=["access_region", "geometry"], geometry="geometry", crs="EPSG:4326")
        if regions.crs is None:
            regions = regions.set_crs(epsg=4326)
        else:
            regions = regions.to_crs(epsg=4326)

        regions["access_region"] = regions["name"].map(region_assignment)
        regions = regions[regions["access_region"].isin(ACCESS_REGION_COLORS)].copy()
        classified_regions = regions[["access_region", "geometry"]].copy()

        gadm_path = resource_dir / "shapes" / "gadm_shapes.geojson"
        if not gadm_path.exists() or classified_regions.empty:
            return classified_regions

        gadm = gpd.read_file(gadm_path)
        if gadm.empty:
            return classified_regions
        if gadm.crs is None:
            gadm = gadm.set_crs(epsg=4326)
        else:
            gadm = gadm.to_crs(epsg=4326)

        bus_union = (
            classified_regions.geometry.union_all()
            if hasattr(classified_regions.geometry, "union_all")
            else classified_regions.geometry.unary_union
        )
        gap_rows = []
        for _, gadm_row in gadm.iterrows():
            gap_geometry = gadm_row.geometry.difference(bus_union)
            if gap_geometry.is_empty:
                continue

            overlap_by_region: dict[str, float] = {}
            for region_name, region_geometry in zip(
                classified_regions["access_region"],
                classified_regions.geometry,
            ):
                overlap_area = gadm_row.geometry.intersection(region_geometry).area
                if overlap_area <= 0.0:
                    continue
                overlap_by_region[str(region_name)] = (
                    overlap_by_region.get(str(region_name), 0.0) + float(overlap_area)
                )

            if overlap_by_region:
                inferred_region = max(overlap_by_region, key=overlap_by_region.get)
            else:
                if classified_regions.empty:
                    continue
                gadm_centroid = gadm_row.geometry.centroid
                nearest_index = min(
                    classified_regions.index,
                    key=lambda idx: classified_regions.at[idx, "geometry"].distance(gadm_centroid),
                )
                inferred_region = str(classified_regions.at[nearest_index, "access_region"])

            gap_rows.append({"access_region": inferred_region, "geometry": gap_geometry})

        if not gap_rows:
            return classified_regions

        gap_regions = gpd.GeoDataFrame(gap_rows, geometry="geometry", crs="EPSG:4326")
        return pd.concat([classified_regions, gap_regions], ignore_index=True)

    demand_by_model = {
        model.key: demand_by_bus_twh(model)
        for model in selected_models
    }
    region_shapes_by_model = {
        model.key: access_region_shapes(model)
        for model in selected_models
    }
    region_demand_by_model = {}
    for model in selected_models:
        demand = demand_by_model[model.key]
        if "access_region" not in demand.columns or demand.empty:
            region_demand_by_model[model.key] = pd.Series(dtype=float)
        else:
            region_demand_by_model[model.key] = (
                demand.dropna(subset=["access_region"])
                .groupby("access_region")["total"]
                .sum()
            )
    global_max_demand = max(
        (
            float(demand["total"].max())
            for demand in demand_by_model.values()
            if not demand.empty
        ),
        default=0.0,
    )
    if global_max_demand <= 0.0:
        return []

    def pie_radius(total_twh: float) -> float:
        """Scale pie area proportionally to annual demand."""
        return max_pie_radius * (float(total_twh) / global_max_demand) ** 0.5

    def format_twh(value: float) -> str:
        """Format TWh/a legend values compactly."""
        if value >= 10.0:
            return f"{value:.0f}"
        if value >= 1.0:
            return f"{value:.1f}"
        return f"{value:.2f}"

    def draw_demand_pie(
        ax: Any,
        x: float,
        y: float,
        industry: float,
        residual_on_grid: float,
        off_grid: float,
    ) -> None:
        """Draw one industry/residual/off-grid demand pie in map coordinates."""
        total = industry + residual_on_grid + off_grid
        if total <= 0.0:
            return
        radius = pie_radius(total)
        current_angle = 90.0
        for value, color in (
            (industry, industry_color),
            (residual_on_grid, residual_color),
            (off_grid, offgrid_color),
        ):
            if value <= 0.0:
                continue
            next_angle = current_angle + 360.0 * value / total
            ax.add_patch(
                Wedge(
                    (x, y),
                    radius,
                    current_angle,
                    next_angle,
                    facecolor=color,
                    edgecolor="white",
                    linewidth=0.55,
                    zorder=5,
                )
            )
            current_angle = next_angle
        ax.add_patch(
            Wedge(
                (x, y),
                radius,
                0.0,
                360.0,
                facecolor="none",
                edgecolor="#111827",
                linewidth=0.55,
                zorder=6,
            )
        )

    def draw_demand_node_map(
        model: Model,
        demand: pd.DataFrame,
        fig: plt.Figure,
        ax: Any,
        show_legend: bool = True,
        panel_label: str | None = None,
    ) -> None:
        """Draw one demand-pie map on a supplied axis."""
        resource_dir = PROJECT_ROOT / "resources" / model.scenario_name
        country = gpd.read_file(resource_dir / "shapes" / "country_shapes.geojson").to_crs(epsg=4326)
        xmin, ymin, xmax, ymax = country.total_bounds
        x_pad = max((xmax - xmin) * RESULT_MAP_PAD_RATIO, 0.02)
        y_pad = max((ymax - ymin) * RESULT_MAP_PAD_RATIO, 0.02)
        plot_xlim = (xmin - x_pad, xmax + x_pad)
        plot_ylim = (ymin - y_pad, ymax + y_pad)

        ax.set_facecolor("white")
        region_shapes = region_shapes_by_model.get(model.key)
        if region_shapes is not None and not region_shapes.empty:
            for region_name, color in ACCESS_REGION_COLORS.items():
                subset = region_shapes[region_shapes["access_region"] == region_name]
                if subset.empty:
                    continue
                subset.plot(
                    ax=ax,
                    color=color,
                    edgecolor=color,
                    linewidth=0.3,
                    alpha=0.12,
                    zorder=0,
                )
        country.boundary.plot(
            ax=ax,
            color="black",
            linewidth=COUNTRY_OUTLINE_LINE_WIDTH,
            zorder=1,
        )

        for bus_name, row in demand.sort_values("total", ascending=False).iterrows():
            draw_demand_pie(
                ax,
                float(row["x"]),
                float(row["y"]),
                float(row["industry"]),
                float(row["residual_on_grid"]),
                float(row["off_grid"]),
            )

        labels = demand[["x", "y"]].copy()
        labels["station_id"] = [station_id_from_bus_name(bus_name) for bus_name in labels.index]
        labels = labels[labels["station_id"].isin(DISPLAY_LABELS)]
        labels = labels[["station_id", "x", "y"]].drop_duplicates("station_id")
        _draw_station_labels(ax, labels, "x", "y", "#1F2933", RESULT_LABEL_SIZE)

        size_values = [
            value
            for value in (global_max_demand / 4.0, global_max_demand / 2.0, global_max_demand)
            if value > 0.0
        ]
        region_totals = region_demand_by_model.get(model.key, pd.Series(dtype=float))
        region_handles = []
        for region_name, color in ACCESS_REGION_COLORS.items():
            total = region_totals.get(region_name, None)
            if total is None or pd.isna(total):
                continue
            region_handles.append(
                Patch(
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.20,
                    label=f"{region_name}: {format_twh(float(total))} TWh/a",
                )
            )
        if region_handles:
            region_legend = ax.legend(
                handles=region_handles,
                loc="upper left",
                frameon=True,
                fontsize=LEGEND_FONT_SIZE,
                title="Regional demand",
                title_fontsize=LEGEND_FONT_SIZE,
                handlelength=1.35,
                handletextpad=0.45,
            )
            ax.add_artist(region_legend)
        if show_legend:
            handles: list[Any] = [
                Patch(facecolor=industry_color, edgecolor="none", label="Industry demand"),
                Patch(facecolor=residual_color, edgecolor="none", label="Residual on-grid demand"),
                Patch(facecolor=offgrid_color, edgecolor="none", label="Off-grid demand"),
            ]
            for value in size_values:
                handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        linestyle="None",
                        markerfacecolor="none",
                        markeredgecolor="#111827",
                        markersize=max(4.0, (pie_radius(value) / max_pie_radius) * 15.0),
                        label=f"{format_twh(value)} TWh/a",
                    )
                )

        ax.set_xlim(plot_xlim)
        ax.set_ylim(plot_ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(FRAME_LINE_WIDTH)
        ax.set_title("")
        if panel_label is not None:
            ax.text(
                0.98,
                0.97,
                panel_label,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=LEGEND_FONT_SIZE,
                color="#111827",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "#d1d5db",
                    "boxstyle": "round,pad=0.22",
                    "alpha": 0.92,
                },
                zorder=10,
            )
        if show_legend:
            ax.legend(
                handles=handles,
                loc="lower left",
                frameon=True,
                fontsize=LEGEND_FONT_SIZE,
                handlelength=1.5,
                columnspacing=0.9,
                handletextpad=0.4,
            )

    output_paths = []
    for model in selected_models:
        fig, ax = plt.subplots(figsize=TWO_COLUMN_EXPORT_FIGSIZE, dpi=FIGURE_DPI)
        ax.set_position([0.0, 0.0, 1.0, 1.0])
        draw_demand_node_map(
            model=model,
            demand=demand_by_model[model.key],
            fig=fig,
            ax=ax,
            show_legend=True,
        )

        output_paths.append(
            _save_figure(
                fig,
                f"electricity_demand_node_map_{model.key}.png",
                fill_canvas=True,
            )
        )

    combined_keys = ("myopic_2025_0", "myopic_2050_no_large_hydro")
    models_by_key = {model.key: model for model in selected_models}
    if all(key in models_by_key for key in combined_keys):
        combined_models = [models_by_key[key] for key in combined_keys]
        n_panels = len(combined_models)
        combined_fig, axes = plt.subplots(
            n_panels,
            1,
            figsize=(TWO_COLUMN_EXPORT_FIGSIZE[0], TWO_COLUMN_EXPORT_FIGSIZE[1] * n_panels),
            dpi=FIGURE_DPI,
        )
        if n_panels == 1:
            axes = [axes]
        panel_height = 1.0 / n_panels
        for index, axis in enumerate(axes):
            axis.set_position([0.0, 1.0 - (index + 1) * panel_height, 1.0, panel_height])

        for index, (axis, model) in enumerate(zip(axes, combined_models)):
            draw_demand_node_map(
                model=model,
                demand=demand_by_model[model.key],
                fig=combined_fig,
                ax=axis,
                show_legend=(index == len(combined_models) - 1),
                panel_label=model.label,
            )

        output_paths.append(
            _save_figure(
                combined_fig,
                "electricity_demand_node_map_myopic_2025_2035_2050_combined.png",
                fill_canvas=True,
                target_size_px=(TWO_COLUMN_IMAGE_SIZE_PX[0], TWO_COLUMN_IMAGE_SIZE_PX[1] * n_panels),
            )
        )

    return output_paths


















# -----------------------------------------------------------
# Demand Coverage
# -----------------------------------------------------------


def plot_electricity_demand_coverage_stack(models: list[Model]) -> Path:
    """Plot separated electricity demand coverage stacks for Elsevier export.

    Parameters
    ----------
    models : list[Model]
        Loaded offgrid models. The function plots all models in the order
        defined by ``MODEL_SPECS``.

    Returns
    -------
    Path
        Absolute path to ``electricity_demand_coverage_stack.png``.
    """
    model_order = {spec["key"]: order for order, spec in enumerate(MODEL_SPECS)}
    selected_models = sorted(models, key=lambda model: model_order[model.key])

    # A compact color set keeps the paper figure calm and readable.
    colors = {
        "PV": "#f2c94c",
        "ROR": "#5ecfc3",
        "Reservoir": "#119c8d",
        "Battery": "#7c3aed",
        "Oil": "#4b5563",
        "Diesel": "#8b5e34",
        "Uncovered": "#d12248",
        "Other": "#94a3b8",
    }
    stack_order = [
        "Reservoir",
        "PV",
        "ROR",
        "Battery",
        "Oil",
        "Diesel",
        "Uncovered",
        "Other",
    ]
    groups = [
        ("on_grid", "On-grid demand"),
        ("off_grid", "Off-grid demand"),
        ("h2", "H2 electrolysis demand"),
    ]

    def normalise_token(value: Any) -> str:
        """Return a lowercase token used for fixed carrier checks."""
        return "".join(char for char in str(value).strip().lower() if char.isalnum())

    def profile_timestep_hours(index: pd.Index) -> float:
        """Infer the time step length of a CSV demand profile in hours."""
        timestamps = pd.to_datetime(index, errors="coerce")
        if pd.isna(timestamps).any():
            return 1.0
        diffs = pd.DatetimeIndex(timestamps).to_series().diff().dropna()
        if diffs.empty:
            return 1.0
        hours = diffs.median().total_seconds() / 3600.0
        return float(hours) if hours > 0.0 else 1.0

    def demand_profile_twh(model: Model, filename: str) -> float:
        """Load one demand profile CSV and return annual demand in TWh."""
        resource_dir = PROJECT_ROOT / "resources" / model.scenario_name
        candidates = [
            resource_dir / f"horizon_{model.year}" / filename,
            resource_dir / filename,
        ]
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if path is None:
            return 0.0
        demand = pd.read_csv(path, index_col=0).apply(pd.to_numeric, errors="raise")
        return float(demand.sum().sum() * profile_timestep_hours(demand.index) / 1e6)

    def carrier_label(carrier: Any, offgrid: bool = False) -> str:
        """Map PyPSA carrier names to compact plot labels."""
        token = normalise_token(carrier)
        if "shed" in token:
            return "Uncovered"
        if "solar" in token or token == "pv":
            return "PV"
        if token in {"ror", "runofriver", "runriver"}:
            return "ROR"
        if "reservoir" in token or token in {"hydro", "hydroreservoir"}:
            return "Reservoir"
        if "battery" in token:
            return "Battery"
        if "diesel" in token:
            return "Diesel"
        if token in {"oil", "ocgt", "ccgt"}:
            return "Oil"
        return "Other"

    def weighted_twh(frame: pd.DataFrame, weights: pd.Series) -> pd.Series:
        """Convert a weighted dispatch table from MW to annual TWh."""
        aligned_weights = pd.to_numeric(
            weights.reindex(frame.index),
            errors="coerce",
        ).fillna(0.0)
        return frame.clip(lower=0.0).multiply(aligned_weights, axis=0).sum() / 1e6

    def h2_electrolysis_demand_twh(network: Any) -> float:
        """Return electricity consumed by H2 electrolysis links in TWh."""
        links = network.links.copy()
        carriers = links.get("carrier", pd.Series("", index=links.index)).map(normalise_token)
        electrolysis_links = links[carriers.str.contains("electrolysis", regex=False)]
        if electrolysis_links.empty:
            return 0.0
        weights = network.snapshot_weightings["objective"]
        link_power = network.links_t.p0.reindex(columns=electrolysis_links.index, fill_value=0.0)
        return float(weighted_twh(link_power, weights).sum())

    def grid_supply_mix_twh(network: Any, demand_twh: float) -> pd.Series:
        """Return grid supply contributions scaled to one demand value."""
        if demand_twh <= 0.0:
            return pd.Series(dtype=float)
        balance = network.statistics.energy_balance().reset_index(name="value")
        balance["value"] = pd.to_numeric(balance["value"], errors="coerce").fillna(0.0)
        component = balance["component"].map(normalise_token)
        carrier = balance["carrier"].map(normalise_token)
        bus_carrier = balance["bus_carrier"].map(normalise_token)

        electricity_bus = bus_carrier.isin({"ac", "lowvoltage"})
        network_carrier = carrier.isin(
            {"ac", "dc", "b2b", "electricitydistributiongrid", "distributiongrid"}
        ) | carrier.str.contains("pipeline", regex=False)
        network_component = component.isin({"line", "lines"}) | (
            component.isin({"link", "links"}) & network_carrier
        )
        shifted_storage = component.isin({"link", "links"}) & carrier.str.contains(
            "discharger",
            regex=False,
        )
        supply = balance[
            (balance["value"] > 0.0)
            & electricity_bus
            & ~component.eq("load")
            & ~network_component
            & ~shifted_storage
        ].copy()
        if supply.empty:
            return pd.Series({"Uncovered": demand_twh})

        supply["label"] = [carrier_label(value) for value in supply["carrier"]]
        contributions = supply.groupby("label")["value"].sum() / 1e6
        total_supply = float(contributions.sum())
        if total_supply <= 0.0:
            return pd.Series({"Uncovered": demand_twh})
        if total_supply >= demand_twh:
            contributions = contributions / total_supply * demand_twh
        else:
            contributions.loc["Uncovered"] = (
                contributions.get("Uncovered", 0.0) + demand_twh - total_supply
            )
        return contributions[contributions > 1e-9]

    def offgrid_supply_mix_twh(model: Model, demand_twh: float) -> pd.Series:
        """Return off-grid supply contributions scaled to off-grid demand."""
        if demand_twh <= 0.0:
            return pd.Series(dtype=float)
        network = model.network
        weights = network.snapshot_weightings["objective"]
        generator_power = network.generators_t.p
        rows = []
        for name, row in network.generators.iterrows():
            bus_name = str(row.get("bus"))
            if bus_name not in network.buses.index or name not in generator_power.columns:
                continue
            bus_carrier = normalise_token(network.buses.at[bus_name, "carrier"])
            if not (bus_name.startswith("offgrid_") or bus_carrier == "offgridac"):
                continue
            energy_twh = float(weighted_twh(generator_power[[name]], weights).iloc[0])
            if energy_twh > 0.0:
                rows.append(
                    {
                        "label": carrier_label(row.get("carrier", ""), offgrid=True),
                        "value": energy_twh,
                    }
                )
        if not rows:
            return pd.Series({"Uncovered": demand_twh})
        contributions = pd.DataFrame(rows).groupby("label")["value"].sum()
        total_supply = float(contributions.sum())
        if total_supply <= 0.0:
            return pd.Series({"Uncovered": demand_twh})
        if total_supply >= demand_twh:
            contributions = contributions / total_supply * demand_twh
        else:
            contributions.loc["Uncovered"] = (
                contributions.get("Uncovered", 0.0) + demand_twh - total_supply
            )
        return contributions[contributions > 1e-9]

    def display_model_label(label: str) -> str:
        """Return the compact scenario label used in the plot."""
        return (
            str(label)
            .replace("2050 0", "2050 zero")
            .replace("2050 23.33", "2050 low")
            .replace("2050 78.33", "2050 mid")
            .replace("2050 133.32", "2050 high")
        )

    def format_value(value: float) -> str:
        """Format segment labels in TWh without repeating the unit."""
        if value < 0.05:
            return "0.0"
        return f"{value:.1f}"

    rows = []
    for model in selected_models:
        # Build three separate demand areas for each selected model.
        on_grid_demand = demand_profile_twh(model, "demand_profiles.csv")
        off_grid_demand = demand_profile_twh(model, "offgrid_demand_profiles.csv")
        show_h2_demand = model.key != "myopic_2050_no_large_hydro"
        h2_demand = h2_electrolysis_demand_twh(model.network) if show_h2_demand else 0.0
        rows.append(
            {
                "label": model.label,
                "on_grid": grid_supply_mix_twh(model.network, on_grid_demand),
                "off_grid": offgrid_supply_mix_twh(model, off_grid_demand),
                "h2": (
                    grid_supply_mix_twh(model.network, h2_demand)
                    if show_h2_demand
                    else pd.Series(dtype=float)
                ),
                "demands": {
                    "on_grid": on_grid_demand,
                    "off_grid": off_grid_demand,
                    "h2": h2_demand,
                },
            }
        )

    # Each demand area receives its own scale, like three independent columns.
    max_by_group = {
        group_key: max(row["demands"][group_key] for row in rows)
        for group_key, _ in groups
    }
    max_by_group = {
        key: value if value > 0.0 else 1.0
        for key, value in max_by_group.items()
    }

    group_width = 1.0
    group_gap = 0.32
    group_starts = {
        group_key: index * (group_width + group_gap)
        for index, (group_key, _) in enumerate(groups)
    }
    x_min = -0.80
    x_max = group_starts[groups[-1][0]] + group_width + 0.48
    y_positions = [0.24 + index * 0.24 for index in reversed(range(len(rows)))]
    bar_height = 0.12
    segment_label_offset = 0.07
    segment_label_min_twh = 15.0
    segment_label_max_count = 3
    demand_font_size = AXIS_FONT_SIZE + 0.7
    header_y = y_positions[0] + 0.22
    row_label_x = -0.07

    fig, ax = plt.subplots(figsize=DEMAND_COVERAGE_FIGSIZE, dpi=FIGURE_DPI)
    ax.set_position([0.012, 0.012, 0.976, 0.976])
    ax.set_facecolor("white")

    # Add compact column headers without vertical divider lines.
    for group_key, group_label in groups:
        start = group_starts[group_key]
        ax.text(
            start + group_width / 2.0,
            header_y,
            f"{group_label}\n[TWh/a]",
            ha="center",
            va="bottom",
            fontsize=demand_font_size,
            color="#111827",
        )

    used_labels = set()
    for y, row in zip(y_positions, rows):
        row_label = display_model_label(row["label"])
        total_demand = sum(float(row["demands"][group_key]) for group_key, _ in groups)
        ax.text(
            row_label_x,
            y + 0.04,
            f"{row_label}\n{format_value(total_demand)} TWh",
            ha="right",
            va="center",
            fontsize=demand_font_size - 0.1,
            linespacing=1.15,
            color="#111827",
        )
        for group_key, _ in groups:
            stack = row[group_key].reindex(stack_order).dropna()
            start = group_starts[group_key]
            left = start
            if stack.empty or row["demands"][group_key] <= 0.0:
                ax.text(
                    start + 0.03,
                    y,
                    "0.0",
                    ha="left",
                    va="center",
                    fontsize=demand_font_size - 0.3,
                    color="#64748b",
                )
                continue
            label_positions = []
            positive_stack = stack[stack > 0.0]
            important_labels = positive_stack[
                positive_stack >= segment_label_min_twh
            ].sort_values(ascending=False)
            important_labels = set(important_labels.head(segment_label_max_count).index)
            for label, value in stack.items():
                if value <= 0.0:
                    continue
                width = float(value) / max_by_group[group_key] * group_width
                color = colors.get(label, colors["Other"])
                ax.barh(
                    y,
                    width,
                    left=left,
                    height=bar_height,
                    color=color,
                    edgecolor="none",
                    linewidth=0.0,
                    zorder=3,
                )
                used_labels.add(label)
                anchor_x = left + width / 2.0
                label_text = format_value(float(value))
                label_is_important = label in important_labels
                if label_is_important:
                    label_x = anchor_x
                    label_y = y - segment_label_offset
                    if width < 0.075:
                        label_x = anchor_x + 0.11
                    label_x = min(max(label_x, start + 0.015), start + group_width + 0.34)
                    label_positions.append((label_x, label_y, anchor_x, label_text))
                left += width
            label_positions.sort(key=lambda item: item[0])
            for index in range(1, len(label_positions)):
                previous = label_positions[index - 1]
                current = label_positions[index]
                minimum_gap = max(0.11, (len(previous[3]) + len(current[3])) * 0.018)
                if current[0] - previous[0] < minimum_gap:
                    label_positions[index] = (
                        previous[0] + minimum_gap,
                        current[1],
                        current[2],
                        current[3],
                    )
            for label_x, label_y, anchor_x, label_text in label_positions:
                ax.text(
                    label_x,
                    label_y,
                    label_text,
                    ha="center",
                    va="top",
                    fontsize=demand_font_size - 1.4,
                    color="#111827",
                    zorder=4,
                )
                ax.plot(
                    [label_x, anchor_x],
                    [label_y + 0.018, y - bar_height / 2.0],
                    color="#475569",
                    linewidth=0.45,
                    alpha=0.8,
                    zorder=2,
                )
            ax.text(
                left + 0.025,
                y,
                format_value(float(row["demands"][group_key])),
                ha="left",
                va="center",
                fontsize=demand_font_size - 0.4,
                color="#111827",
            )

    # Keep the legend compact and in the requested upper-right position.
    legend_handles = [
        Patch(facecolor=colors[label], edgecolor="white", label=label)
        for label in stack_order
        if label in used_labels
    ]
    if legend_handles:
        ax.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.012),
            ncol=len(legend_handles),
            frameon=False,
            fontsize=demand_font_size - 0.2,
            columnspacing=1.1,
            handlelength=1.0,
            handletextpad=0.35,
        )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.08, header_y + 0.20)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#111827")
        spine.set_linewidth(FRAME_LINE_WIDTH)
    return _save_figure(
        fig,
        "electricity_demand_coverage_stack.png",
        fill_canvas=True,
        target_size_px=DEMAND_COVERAGE_IMAGE_SIZE_PX,
    )


# -----------------------------------------------------------
# System Costs
# -----------------------------------------------------------


def plot_system_costs_capex_opex_stack(models: list[Model]) -> Path:
    """Plot annual system costs split into CAPEX and OPEX stacks.

    Parameters
    ----------
    models : list[Model]
        Loaded offgrid models. The function plots all models in the order
        defined by ``MODEL_SPECS``.

    Returns
    -------
    Path
        Absolute path to ``system_costs_capex_opex_stack.png``.
    """
    model_order = {spec["key"]: order for order, spec in enumerate(MODEL_SPECS)}
    selected_models = sorted(models, key=lambda model: model_order[model.key])

    colors = {
        "On-grid PV": "#f2c94c",
        "Off-grid PV": "#f59e0b",
        "ROR": "#5ecfc3",
        "Reservoir": "#119c8d",
        "Battery": "#7c3aed",
        "H2 electrolysis": "#60a5fa",
        "H2 storage": "#2563eb",
        "H2 network": "#1d4ed8",
        "Transmission": "#f97316",
        "Oil": "#4b5563",
        "Diesel": "#8b5e34",
        "Gas": "#64748b",
        "Coal": "#334155",
        "Load shedding": "#d12248",
        "Other": "#94a3b8",
    }
    stack_order = [
        "Reservoir",
        "On-grid PV",
        "Off-grid PV",
        "ROR",
        "Battery",
        "H2 electrolysis",
        "H2 storage",
        "H2 network",
        "Transmission",
        "Oil",
        "Diesel",
        "Gas",
        "Coal",
        "Load shedding",
        "Other",
    ]
    cost_groups = [("capex", "CAPEX"), ("opex", "OPEX")]

    def normalise_token(value: Any) -> str:
        """Return a lowercase token used for fixed carrier checks."""
        return "".join(char for char in str(value).strip().lower() if char.isalnum())

    def display_model_label(label: str) -> str:
        """Return the compact scenario label used in the plot."""
        return (
            str(label)
            .replace("2050 0", "2050 zero")
            .replace("2050 23.33", "2050 low")
            .replace("2050 78.33", "2050 mid")
            .replace("2050 133.32", "2050 high")
        )

    def cost_label(component: Any, carrier: Any) -> str:
        """Map PyPSA component/carrier pairs to compact technology groups."""
        component_token = normalise_token(component)
        carrier_token = normalise_token(carrier)

        if "loadshedding" in carrier_token or "shed" in carrier_token:
            return "Load shedding"
        if component_token in {"line", "lines"}:
            return "Transmission"
        if component_token in {"link", "links"} and carrier_token in {"dc", "b2b"}:
            return "Transmission"
        if "pipeline" in carrier_token:
            return "H2 network"
        if "electrolysis" in carrier_token:
            return "H2 electrolysis"
        if "h2store" in carrier_token or (
            component_token in {"store", "stores"} and "h2" in carrier_token
        ):
            return "H2 storage"
        if carrier_token == "h2":
            return "H2 network"
        if "battery" in carrier_token:
            return "Battery"
        if carrier_token in {"ror", "runofriver", "runriver"}:
            return "ROR"
        if "reservoir" in carrier_token or "hydro" in carrier_token:
            return "Reservoir"
        if "diesel" in carrier_token:
            return "Diesel"
        if "oil" in carrier_token:
            return "Oil"
        if carrier_token in {"gas", "ocgt", "ccgt"}:
            return "Gas"
        if carrier_token in {"coal", "lignite"}:
            return "Coal"
        return "Other"

    def is_solar_generator(component: Any, carrier: Any) -> bool:
        """Return true for solar generator entries handled separately."""
        component_token = normalise_token(component)
        carrier_token = normalise_token(carrier)
        return component_token in {"generator", "generators"} and (
            "solar" in carrier_token or carrier_token == "pv"
        )

    def is_offgrid_bus(network: Any, bus_name: Any) -> bool:
        """Return true if a bus belongs to the explicit off-grid subsystem."""
        bus_text = str(bus_name)
        if bus_text.startswith("offgrid_"):
            return True
        if bus_text not in network.buses.index:
            return False
        return normalise_token(network.buses.at[bus_text, "carrier"]) == "offgridac"

    def statistics_costs_bn(network: Any, cost_type: str) -> pd.Series:
        """Return one CAPEX/OPEX cost stack in bn EUR/a."""
        if cost_type == "capex":
            costs = network.statistics.capex()
        elif cost_type == "opex":
            costs = network.statistics.opex()
        else:
            raise ValueError(f"Unsupported cost type: {cost_type}")

        rows = []
        for index, value in costs.items():
            if isinstance(index, tuple):
                component, carrier = index[0], index[1]
            else:
                component, carrier = "", index
            value = float(value)
            if value <= 0.0 or is_solar_generator(component, carrier):
                continue
            rows.append(
                {
                    "label": cost_label(component, carrier),
                    "value": value / 1e9,
                }
            )
        if not rows:
            return pd.Series(dtype=float)
        return pd.DataFrame(rows).groupby("label")["value"].sum()

    def solar_generator_costs_bn(network: Any, cost_type: str) -> pd.Series:
        """Return solar generator costs split into on-grid and off-grid PV."""
        generators = network.generators.copy()
        if generators.empty:
            return pd.Series(dtype=float)

        solar = generators[
            generators["carrier"].map(normalise_token).str.contains("solar", regex=False)
            | generators["carrier"].map(normalise_token).eq("pv")
        ].copy()
        if solar.empty:
            return pd.Series(dtype=float)

        if cost_type == "capex":
            capacity = pd.to_numeric(solar.get("p_nom_opt", solar["p_nom"]), errors="coerce")
            costs = capacity.fillna(0.0) * pd.to_numeric(
                solar["capital_cost"], errors="coerce"
            ).fillna(0.0)
        elif cost_type == "opex":
            weights = network.snapshot_weightings["objective"]
            dispatch = network.generators_t.p.reindex(columns=solar.index, fill_value=0.0)
            weighted_energy = dispatch.clip(lower=0.0).multiply(weights, axis=0).sum()
            costs = weighted_energy * pd.to_numeric(
                solar["marginal_cost"], errors="coerce"
            ).fillna(0.0)
        else:
            raise ValueError(f"Unsupported cost type: {cost_type}")

        labels = solar["bus"].map(
            lambda bus_name: "Off-grid PV" if is_offgrid_bus(network, bus_name) else "On-grid PV"
        )
        result = costs.groupby(labels).sum() / 1e9
        return result[result > 0.0]

    def format_cost(value: float) -> str:
        """Format annual costs in bn EUR/a."""
        if value < 0.005:
            return "<0.01"
        if value < 1.0:
            return f"{value:.2f}"
        return f"{value:.1f}"

    rows = []
    for model in selected_models:
        capex = statistics_costs_bn(model.network, "capex").add(
            solar_generator_costs_bn(model.network, "capex"), fill_value=0.0
        )
        opex = statistics_costs_bn(model.network, "opex").add(
            solar_generator_costs_bn(model.network, "opex"), fill_value=0.0
        )
        rows.append(
            {
                "label": model.label,
                "capex": capex,
                "opex": opex,
                "totals": {
                    "capex": float(capex.sum()),
                    "opex": float(opex.sum()),
                },
            }
        )

    max_by_group = {
        group_key: max(row["totals"][group_key] for row in rows)
        for group_key, _ in cost_groups
    }
    max_by_group = {
        key: value if value > 0.0 else 1.0
        for key, value in max_by_group.items()
    }

    group_width = 1.0
    group_gap = 0.38
    group_starts = {
        group_key: index * (group_width + group_gap)
        for index, (group_key, _) in enumerate(cost_groups)
    }
    x_min = -0.80
    x_max = group_starts[cost_groups[-1][0]] + group_width + 0.50
    y_positions = [0.48 + index * 0.24 for index in reversed(range(len(rows)))]
    bar_height = 0.12
    cost_font_size = AXIS_FONT_SIZE + 0.7
    segment_label_offset = 0.07
    segment_label_min_bn = 10.00
    segment_label_max_count = 3
    header_y = y_positions[0] + 0.16
    row_label_x = -0.07

    fig, ax = plt.subplots(figsize=DEMAND_COVERAGE_FIGSIZE, dpi=FIGURE_DPI)
    ax.set_position([0.012, 0.012, 0.976, 0.976])
    ax.set_facecolor("white")

    for group_key, group_label in cost_groups:
        start = group_starts[group_key]
        ax.text(
            start + group_width / 2.0,
            header_y,
            f"{group_label}\n[bn EUR/a]",
            ha="center",
            va="bottom",
            fontsize=cost_font_size,
            color="#111827",
        )

    cost_value_rows = []
    for row in rows:
        for group_key, _ in cost_groups:
            for label, value in row[group_key].items():
                if value > 0.0:
                    cost_value_rows.append(
                        {
                            "model_label": display_model_label(row["label"]),
                            "cost_group": group_key,
                            "technology": label,
                            "value_bn_eur_a": float(value),
                        }
                    )
            cost_value_rows.append(
                {
                    "model_label": display_model_label(row["label"]),
                    "cost_group": group_key,
                    "technology": "Total",
                    "value_bn_eur_a": row["totals"][group_key],
                }
            )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cost_value_rows).to_csv(
        OUTPUT_DIR / "system_costs_capex_opex_stack_values.csv",
        index=False,
    )

    used_labels = set()
    for y, row in zip(y_positions, rows):
        row_label = display_model_label(row["label"])
        total_cost = row["totals"]["capex"] + row["totals"]["opex"]
        ax.text(
            row_label_x,
            y + 0.04,
            f"{row_label}\n{format_cost(total_cost)} bn EUR/a",
            ha="right",
            va="center",
            fontsize=cost_font_size - 0.1,
            linespacing=1.15,
            color="#111827",
        )

        for group_key, _ in cost_groups:
            stack = row[group_key].reindex(stack_order).dropna()
            start = group_starts[group_key]
            left = start
            if stack.empty or row["totals"][group_key] <= 0.0:
                ax.text(
                    start + 0.03,
                    y,
                    "0.00",
                    ha="left",
                    va="center",
                    fontsize=cost_font_size - 0.3,
                    color="#64748b",
                )
                continue

            label_positions = []
            positive_stack = stack[stack > 0.0]
            important_labels = positive_stack[
                positive_stack >= segment_label_min_bn
            ].sort_values(ascending=False)
            important_labels = set(important_labels.head(segment_label_max_count).index)
            for label, value in stack.items():
                if value <= 0.0:
                    continue
                width = float(value) / max_by_group[group_key] * group_width
                color = colors.get(label, colors["Other"])
                ax.barh(
                    y,
                    width,
                    left=left,
                    height=bar_height,
                    color=color,
                    edgecolor="none",
                    linewidth=0.0,
                    zorder=3,
                )
                used_labels.add(label)
                anchor_x = left + width / 2.0
                label_text = f"{format_cost(float(value))}bn"
                if label in important_labels:
                    label_x = anchor_x
                    label_y = y - segment_label_offset
                    if width < 0.075:
                        label_x = anchor_x + 0.11
                    label_x = min(max(label_x, start + 0.015), start + group_width + 0.34)
                    label_positions.append((label_x, label_y, anchor_x, label_text))
                left += width

            label_positions.sort(key=lambda item: item[0])
            for index in range(1, len(label_positions)):
                previous = label_positions[index - 1]
                current = label_positions[index]
                minimum_gap = max(0.11, (len(previous[3]) + len(current[3])) * 0.018)
                if current[0] - previous[0] < minimum_gap:
                    label_positions[index] = (
                        previous[0] + minimum_gap,
                        current[1],
                        current[2],
                        current[3],
                    )
            for label_x, label_y, anchor_x, label_text in label_positions:
                ax.text(
                    label_x,
                    label_y,
                    label_text,
                    ha="center",
                    va="top",
                    fontsize=cost_font_size - 1.4,
                    color="#111827",
                    zorder=4,
                )
                ax.plot(
                    [label_x, anchor_x],
                    [label_y + 0.018, y - bar_height / 2.0],
                    color="#475569",
                    linewidth=0.45,
                    alpha=0.8,
                    zorder=2,
                )
            ax.text(
                left + 0.025,
                y,
                format_cost(float(row["totals"][group_key])),
                ha="left",
                va="center",
                fontsize=cost_font_size - 0.4,
                color="#111827",
            )

    legend_handles = [
        Patch(facecolor=colors[label], edgecolor="white", label=label)
        for label in stack_order
        if label in used_labels
    ]
    if legend_handles:
        ax.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.002),
            ncol=min(6, len(legend_handles)),
            frameon=False,
            fontsize=cost_font_size - 0.7,
            columnspacing=0.9,
            handlelength=1.0,
            handletextpad=0.35,
        )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.08, header_y + 0.28)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#111827")
        spine.set_linewidth(FRAME_LINE_WIDTH)
    return _save_figure(
        fig,
        "system_costs_capex_opex_stack.png",
        fill_canvas=True,
        target_size_px=DEMAND_COVERAGE_IMAGE_SIZE_PX,
    )


# -----------------------------------------------------------
# Levelized Cost of Hydrogen
# -----------------------------------------------------------

def plot_dual_price_lcoh_comparison(models: list[Model]) -> Path:
    """Calculate and plot system-level LCOH using nodal electricity prices.

    Electricity consumed by electrolyzers is priced with the marginal price at
    the corresponding electricity bus. Direct H2 technology and infrastructure
    costs are added separately.

    Parameters
    ----------
    models : list[Model]
        Loaded offgrid models. The function compares the non-zero 2050 H2
        export scenarios.

    Returns
    -------
    Path
        Absolute path to ``h2_dual_price_lcoh_comparison.png``.
    """
    export_keys = (
        "myopic_2050_23p33",
        "myopic_2050_78p33",
        "myopic_2050_133p32",
    )
    models_by_key = {model.key: model for model in models}
    selected_models = [
        models_by_key[key]
        for key in export_keys
        if key in models_by_key
    ]
    if not selected_models:
        raise ValueError("No non-zero 2050 H2 export models available for LCOH plot.")

    def normalise_token(value: Any) -> str:
        """Return a lowercase token used for carrier checks."""
        return "".join(char for char in str(value).strip().lower() if char.isalnum())

    def display_model_label(label: str) -> str:
        """Return the compact scenario label used in the plot."""
        return (
            str(label)
            .replace("2050 23.33", "2050 low")
            .replace("2050 78.33", "2050 mid")
            .replace("2050 133.32", "2050 high")
        )

    def weighted_load_mwh(network: Any, load_index: pd.Index) -> float:
        """Return annual weighted load in MWh/a."""
        if len(load_index) == 0:
            return 0.0
        weights = pd.to_numeric(
            network.snapshot_weightings["objective"],
            errors="coerce",
        ).fillna(0.0)
        if hasattr(network.loads_t, "p_set") and not network.loads_t.p_set.empty:
            load_profile = network.loads_t.p_set.reindex(
                columns=load_index,
                fill_value=0.0,
            ).sum(axis=1)
            return float(load_profile.reindex(weights.index).fillna(0.0).mul(weights).sum())

        static_load = pd.to_numeric(
            network.loads.reindex(load_index).get(
                "p_set",
                pd.Series(0.0, index=load_index),
            ),
            errors="coerce",
        ).fillna(0.0)
        return float(static_load.sum() * weights.sum())

    def h2_export_load_mwh(network: Any) -> float:
        """Return annual served H2 export load in MWh H2/a."""
        loads = network.loads.copy()
        if loads.empty or "bus" not in loads.columns:
            return 0.0
        carriers = loads.get("carrier", pd.Series("", index=loads.index)).astype(str)
        h2_export_load = loads["bus"].astype(str).eq("H2 export bus") | (
            carriers.str.lower().str.contains("h2", regex=False)
            & carriers.str.lower().str.contains("export", regex=False)
        )
        export_demand_mwh = weighted_load_mwh(network, loads.index[h2_export_load])

        load_shedding_mwh = 0.0
        generators = network.generators.copy()
        if (
            not generators.empty
            and "bus" in generators.columns
            and hasattr(network.generators_t, "p")
            and not network.generators_t.p.empty
        ):
            generator_carriers = generators.get(
                "carrier",
                pd.Series("", index=generators.index),
            ).map(normalise_token)
            h2_export_shedding = generators["bus"].astype(str).eq("H2 export bus") & (
                generator_carriers.str.contains("loadshedding", regex=False)
            )
            shedding_index = generators.index[h2_export_shedding]
            if len(shedding_index) > 0:
                weights = pd.to_numeric(
                    network.snapshot_weightings["objective"],
                    errors="coerce",
                ).fillna(0.0)
                shedding_dispatch = network.generators_t.p.reindex(
                    columns=shedding_index,
                    fill_value=0.0,
                )
                shedding_dispatch = shedding_dispatch.clip(lower=0.0).reindex(
                    index=weights.index,
                ).fillna(0.0)
                load_shedding_mwh = float(
                    shedding_dispatch.sum(axis=1).mul(weights).sum()
                )

        return max(export_demand_mwh - load_shedding_mwh, 0.0)

    def electrolysis_links(network: Any) -> pd.DataFrame:
        """Return H2 electrolysis links."""
        links = network.links.copy()
        if links.empty:
            return links
        carriers = links.get("carrier", pd.Series("", index=links.index)).map(normalise_token)
        return links[carriers.str.contains("electrolysis", regex=False)].copy()

    def electrolysis_electricity_cost_eur(network: Any) -> tuple[float, float]:
        """Return nodal-price electricity cost and input for electrolysis."""
        links = electrolysis_links(network)
        if links.empty:
            return 0.0, 0.0
        marginal_prices = network.buses_t.marginal_price
        if marginal_prices.empty:
            raise ValueError("Network does not contain bus marginal prices.")

        missing_buses = sorted(set(links["bus0"]).difference(marginal_prices.columns))
        if missing_buses:
            missing = ", ".join(missing_buses)
            raise ValueError(f"Missing marginal prices for electrolysis bus0 entries: {missing}")

        weights = pd.to_numeric(
            network.snapshot_weightings["objective"],
            errors="coerce",
        ).fillna(0.0)
        dispatch = network.links_t.p0.reindex(
            columns=links.index,
            fill_value=0.0,
        ).clip(lower=0.0)
        dispatch = dispatch.reindex(index=weights.index).fillna(0.0)

        electricity_cost = 0.0
        for bus_name, bus_links in links.groupby("bus0").groups.items():
            bus_dispatch = dispatch.reindex(columns=bus_links, fill_value=0.0).sum(axis=1)
            bus_price = pd.to_numeric(
                marginal_prices[bus_name],
                errors="coerce",
            ).reindex(weights.index).fillna(0.0)
            electricity_cost += float(bus_dispatch.mul(bus_price).mul(weights).sum())

        electricity_input = float(dispatch.sum(axis=1).mul(weights).sum())
        return electricity_cost, electricity_input

    def h2_production_mwh(network: Any) -> float:
        """Return annual H2 output from electrolysis in MWh H2/a."""
        links = electrolysis_links(network)
        if links.empty or not hasattr(network.links_t, "p1") or network.links_t.p1.empty:
            return 0.0
        weights = pd.to_numeric(
            network.snapshot_weightings["objective"],
            errors="coerce",
        ).fillna(0.0)
        production = -network.links_t.p1.reindex(
            columns=links.index,
            fill_value=0.0,
        )
        production = production.clip(lower=0.0).reindex(index=weights.index).fillna(0.0)
        return float(production.sum(axis=1).mul(weights).sum())

    def h2_direct_costs_eur(network: Any) -> dict[str, float]:
        """Return direct H2 technology and infrastructure costs in EUR/a."""
        capex = pd.to_numeric(network.statistics.capex(), errors="coerce").fillna(0.0)
        opex = pd.to_numeric(network.statistics.opex(), errors="coerce").fillna(0.0)
        result = {
            "c_tech_capex_eur_a": 0.0,
            "c_tech_opex_eur_a": 0.0,
            "c_infra_capex_eur_a": 0.0,
            "c_infra_opex_eur_a": 0.0,
        }

        def component_bucket(index: Any) -> str | None:
            if isinstance(index, tuple):
                carrier = index[1]
            else:
                carrier = index
            carrier_token = normalise_token(carrier)
            if "electrolysis" in carrier_token:
                return "tech"
            if (
                "pipeline" in carrier_token
                or "h2store" in carrier_token
                or ("h2" in carrier_token and "export" in carrier_token)
                or carrier_token == "h2"
                or "hydrogen" in carrier_token
            ):
                return "infra"
            return None

        for index, value in capex.items():
            bucket = component_bucket(index)
            if bucket == "tech":
                result["c_tech_capex_eur_a"] += max(float(value), 0.0)
            elif bucket == "infra":
                result["c_infra_capex_eur_a"] += max(float(value), 0.0)

        for index, value in opex.items():
            bucket = component_bucket(index)
            if bucket == "tech":
                result["c_tech_opex_eur_a"] += max(float(value), 0.0)
            elif bucket == "infra":
                result["c_infra_opex_eur_a"] += max(float(value), 0.0)

        return result

    rows = []
    for model in selected_models:
        network = model.network
        c_elec_eur, electrolysis_input_mwh = electrolysis_electricity_cost_eur(network)
        direct_costs = h2_direct_costs_eur(network)
        c_tech_eur = direct_costs["c_tech_capex_eur_a"] + direct_costs["c_tech_opex_eur_a"]
        c_infra_eur = direct_costs["c_infra_capex_eur_a"] + direct_costs["c_infra_opex_eur_a"]
        lcoh_cost_eur = c_elec_eur + c_tech_eur + c_infra_eur
        h2_export_mwh = h2_export_load_mwh(network)
        h2_production = h2_production_mwh(network)
        h2_export_kg = h2_export_mwh * 1000.0 / H2_LHV_KWH_PER_KG

        if h2_export_mwh <= 0.0 or h2_export_kg <= 0.0:
            lcoh_eur_per_mwh = float("nan")
            lcoh_eur_per_kg = float("nan")
            c_elec_eur_per_kg = float("nan")
            c_tech_eur_per_kg = float("nan")
            c_infra_eur_per_kg = float("nan")
        else:
            lcoh_eur_per_mwh = lcoh_cost_eur / h2_export_mwh
            lcoh_eur_per_kg = lcoh_cost_eur / h2_export_kg
            c_elec_eur_per_kg = c_elec_eur / h2_export_kg
            c_tech_eur_per_kg = c_tech_eur / h2_export_kg
            c_infra_eur_per_kg = c_infra_eur / h2_export_kg

        rows.append(
            {
                "model_key": model.key,
                "label": display_model_label(model.label),
                "h2_export_twh_a": h2_export_mwh / 1e6,
                "h2_production_twh_a": h2_production / 1e6,
                "h2_export_million_tonnes_a": h2_export_kg / 1e9,
                "electrolysis_electricity_twh_a": electrolysis_input_mwh / 1e6,
                "weighted_electricity_price_eur_per_mwh_el": (
                    c_elec_eur / electrolysis_input_mwh
                    if electrolysis_input_mwh > 0.0
                    else float("nan")
                ),
                "c_elec_bn_eur_a": c_elec_eur / 1e9,
                "c_tech_capex_bn_eur_a": direct_costs["c_tech_capex_eur_a"] / 1e9,
                "c_tech_opex_bn_eur_a": direct_costs["c_tech_opex_eur_a"] / 1e9,
                "c_tech_bn_eur_a": c_tech_eur / 1e9,
                "c_infra_capex_bn_eur_a": direct_costs["c_infra_capex_eur_a"] / 1e9,
                "c_infra_opex_bn_eur_a": direct_costs["c_infra_opex_eur_a"] / 1e9,
                "c_infra_bn_eur_a": c_infra_eur / 1e9,
                "lcoh_cost_bn_eur_a": lcoh_cost_eur / 1e9,
                "lcoh_electricity_component_eur_per_kg_h2": c_elec_eur_per_kg,
                "lcoh_tech_component_eur_per_kg_h2": c_tech_eur_per_kg,
                "lcoh_infra_component_eur_per_kg_h2": c_infra_eur_per_kg,
                "lcoh_dual_price_eur_per_mwh_h2": lcoh_eur_per_mwh,
                "lcoh_dual_price_eur_per_kg_h2": lcoh_eur_per_kg,
                "h2_lhv_kwh_per_kg": H2_LHV_KWH_PER_KG,
            }
        )

    table = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_DIR / "h2_dual_price_lcoh_values.csv", index=False)

    max_lcoh = float(table["lcoh_dual_price_eur_per_kg_h2"].max())
    max_lcoh = max_lcoh if max_lcoh > 0.0 else 1.0
    max_export = float(table["h2_export_twh_a"].max())

    def format_lcoh(value: float) -> str:
        """Format LCOH in EUR/kg H2."""
        return f"{value:.2f}"

    def format_twh(value: float) -> str:
        """Format annual H2 exports in TWh/a."""
        return f"{value:.2f}"

    def format_price(value: float) -> str:
        """Format electricity price in EUR/MWh."""
        return f"{value:.1f}"

    y_positions = [0.32 + index * 0.26 for index in reversed(range(len(table)))]
    label_font_size = AXIS_FONT_SIZE + 0.7
    bar_height = 0.13
    bar_start = 0.24
    bar_width = 1.0
    export_x = bar_start + bar_width + 0.33
    price_x = export_x + 0.50
    x_min = -0.18
    x_max = price_x + 0.46
    header_y = y_positions[0] + 0.24

    fig, ax = plt.subplots(figsize=DEMAND_COVERAGE_FIGSIZE, dpi=FIGURE_DPI)
    ax.set_position([0.012, 0.012, 0.976, 0.976])
    ax.set_facecolor("white")

    ax.text(
        bar_start + bar_width / 2.0,
        header_y,
        "Dual-price LCOH\n[EUR/kg H2]",
        ha="center",
        va="bottom",
        fontsize=label_font_size,
        color="#111827",
    )
    ax.text(
        export_x,
        header_y,
        "H2 export\n[TWh/a]",
        ha="center",
        va="bottom",
        fontsize=label_font_size,
        color="#111827",
    )
    ax.text(
        price_x,
        header_y,
        "Electricity price\n[EUR/MWh el]",
        ha="center",
        va="bottom",
        fontsize=label_font_size,
        color="#111827",
    )

    electricity_color = "#14a38b"
    tech_color = "#60a5fa"
    infra_color = "#d97706"
    bar_bg_color = "#e5e7eb"
    for y, (_, row) in zip(y_positions, table.iterrows()):
        ax.text(
            0.12,
            y,
            row["label"],
            ha="right",
            va="center",
            fontsize=label_font_size,
            color="#111827",
        )
        ax.barh(
            y,
            bar_width,
            left=bar_start,
            height=bar_height,
            color=bar_bg_color,
            edgecolor="none",
            zorder=1,
        )

        left = bar_start
        for column, color in (
            ("lcoh_electricity_component_eur_per_kg_h2", electricity_color),
            ("lcoh_tech_component_eur_per_kg_h2", tech_color),
            ("lcoh_infra_component_eur_per_kg_h2", infra_color),
        ):
            value = float(row[column])
            width = value / max_lcoh * bar_width
            ax.barh(
                y,
                width,
                left=left,
                height=bar_height,
                color=color,
                edgecolor="none",
                zorder=3,
            )
            left += width

        ax.text(
            left + 0.03,
            y,
            format_lcoh(float(row["lcoh_dual_price_eur_per_kg_h2"])),
            ha="left",
            va="center",
            fontsize=label_font_size - 0.2,
            color="#111827",
        )

        export_marker_size = 35 + 105 * float(row["h2_export_twh_a"]) / max_export
        ax.scatter(
            [export_x],
            [y],
            s=export_marker_size,
            marker="o",
            color=electricity_color,
            edgecolors="white",
            linewidths=0.7,
            zorder=3,
        )
        ax.text(
            export_x + 0.11,
            y,
            format_twh(float(row["h2_export_twh_a"])),
            ha="left",
            va="center",
            fontsize=label_font_size - 0.2,
            color="#111827",
        )

        ax.text(
            price_x,
            y,
            format_price(float(row["weighted_electricity_price_eur_per_mwh_el"])),
            ha="center",
            va="center",
            fontsize=label_font_size - 0.2,
            color="#111827",
        )

    ax.text(
        bar_start,
        0.06,
        f"Electricity priced with nodal marginal prices; "
        f"LHV = {H2_LHV_KWH_PER_KG:.3f} kWh/kg H2",
        ha="left",
        va="bottom",
        fontsize=label_font_size - 1.1,
        color="#475569",
    )

    handles = [
        Patch(facecolor=electricity_color, edgecolor="none", label="Electricity procurement"),
        Patch(facecolor=tech_color, edgecolor="none", label="H2 technology"),
        Patch(facecolor=infra_color, edgecolor="none", label="H2 infrastructure"),
    ]
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.002),
        ncol=3,
        frameon=False,
        fontsize=label_font_size - 0.8,
        columnspacing=1.0,
        handlelength=1.0,
        handletextpad=0.35,
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.06, header_y + 0.22)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#111827")
        spine.set_linewidth(FRAME_LINE_WIDTH)

    return _save_figure(
        fig,
        "h2_dual_price_lcoh_comparison.png",
        fill_canvas=True,
        target_size_px=DEMAND_COVERAGE_IMAGE_SIZE_PX,
    )

# -----------------------------------------------------------
# Installed Capacity
# -----------------------------------------------------------


def plot_installed_capacity_by_carrier(models: list[Model]) -> Path:
    """Plot installed electricity capacity against modelled expansion limits.

    Parameters
    ----------
    models : list[Model]
        Loaded offgrid models. The function plots all models in the order
        defined by ``MODEL_SPECS``.

    Returns
    -------
    Path
        Absolute path to ``electricity_installed_capacity_by_carrier.png``.
    """
    model_order = {spec["key"]: order for order, spec in enumerate(MODEL_SPECS)}
    selected_models = sorted(models, key=lambda model: model_order[model.key])

    # Paper-style colors, with grey used for the available total capacity potential.
    colors = {
        "Reservoir": "#119c8d",
        "ROR": "#5ecfc3",
        "On-grid PV": "#f2c94c",
        "Off-grid PV": "#f6a623",
        "Potential": "#e5e7eb",
    }
    groups = [
        ("reservoir", "Reservoir"),
        ("ror", "ROR"),
        ("pv", "PV"),
    ]

    def normalise_token(value: Any) -> str:
        """Return a lowercase token used for carrier and bus checks."""
        return "".join(char for char in str(value).strip().lower() if char.isalnum())

    def capacity_column(frame: pd.DataFrame, primary: str, fallback: str) -> pd.Series:
        """Return a numeric capacity column, using a fallback when needed."""
        if primary in frame.columns:
            values = pd.to_numeric(frame[primary], errors="coerce").fillna(0.0)
            if values.sum() > 0.0:
                return values
        if fallback in frame.columns:
            return pd.to_numeric(frame[fallback], errors="coerce").fillna(0.0)
        return pd.Series(0.0, index=frame.index)

    def finite_capacity_limit(frame: pd.DataFrame, installed_mw: float) -> float:
        """Return finite p_nom_max plus fixed capacity without finite p_nom_max."""
        if frame.empty:
            return installed_mw

        installed = capacity_column(frame, "p_nom_opt", "p_nom")
        if "p_nom_max" not in frame.columns:
            return max(installed_mw, float(installed.sum()))

        limits = pd.to_numeric(frame["p_nom_max"], errors="coerce")
        limits = limits.replace([float("inf"), -float("inf")], pd.NA)
        finite_limit = limits.notna() & (limits > 0.0)
        finite_sum = float(limits[finite_limit].sum())
        fixed_without_limit = float(installed[~finite_limit].sum())
        return max(finite_sum + fixed_without_limit, installed_mw)

    def is_offgrid_bus(network: Any, bus_name: Any) -> bool:
        """Return true when a bus belongs to the off-grid module."""
        bus_text = str(bus_name)
        if bus_text.startswith("offgrid_"):
            return True
        if bus_text not in network.buses.index:
            return False
        return normalise_token(network.buses.at[bus_text, "carrier"]) == "offgridac"

    def display_model_label(label: str) -> str:
        """Return the compact scenario label used in the plot."""
        return (
            str(label)
            .replace("2050 0", "2050 zero")
            .replace("2050 23.33", "2050 low")
            .replace("2050 78.33", "2050 mid")
            .replace("2050 133.32", "2050 high")
        )

    def format_gw(value_mw: float) -> str:
        """Format MW values as GW with one decimal place."""
        return f"{value_mw / 1000.0:.1f}"

    rows = []
    for model in selected_models:
        network = model.network
        generators = network.generators.copy()
        storage_units = network.storage_units.copy()

        # Normalize carrier names once before filtering component tables.
        generators["carrier_token"] = generators["carrier"].map(normalise_token)
        storage_units["carrier_token"] = storage_units["carrier"].map(normalise_token)

        pv_generators = generators[
            generators["carrier_token"].str.contains("solar", regex=False)
            | generators["carrier_token"].eq("pv")
        ].copy()
        pv_generators["is_offgrid"] = [
            is_offgrid_bus(network, bus_name) for bus_name in pv_generators["bus"]
        ]
        ongrid_pv = pv_generators[~pv_generators["is_offgrid"]]
        offgrid_pv = pv_generators[pv_generators["is_offgrid"]]

        ror_generators = generators[
            generators["carrier_token"].isin({"ror", "runofriver", "runriver"})
        ].copy()
        reservoir_units = storage_units[
            storage_units["carrier_token"].str.contains("hydro", regex=False)
            | storage_units["carrier_token"].str.contains("reservoir", regex=False)
        ].copy()

        reservoir_installed = float(capacity_column(reservoir_units, "p_nom_opt", "p_nom").sum())
        ror_installed = float(capacity_column(ror_generators, "p_nom_opt", "p_nom").sum())
        ongrid_pv_installed = float(capacity_column(ongrid_pv, "p_nom_opt", "p_nom").sum())
        offgrid_pv_installed = float(capacity_column(offgrid_pv, "p_nom_opt", "p_nom").sum())

        # The PV expansion limit is the finite on-grid p_nom_max from the
        # renewable-potential workflow. Off-grid PV has no finite p_nom_max in
        # these result files and is therefore only shown as installed capacity.
        pv_limit = finite_capacity_limit(ongrid_pv, ongrid_pv_installed)

        rows.append(
            {
                "label": model.label,
                "installed": {
                    "reservoir": reservoir_installed,
                    "ror": ror_installed,
                    "pv_ongrid": ongrid_pv_installed,
                    "pv_offgrid": offgrid_pv_installed,
                },
                "limit": {
                    "reservoir": finite_capacity_limit(reservoir_units, reservoir_installed),
                    "ror": finite_capacity_limit(ror_generators, ror_installed),
                    "pv": pv_limit,
                },
            }
        )

    # Each carrier column receives its own scale, so all three columns remain readable.
    max_by_group = {
        "reservoir": max(row["limit"]["reservoir"] for row in rows),
        "ror": max(row["limit"]["ror"] for row in rows),
        "pv": max(row["limit"]["pv"] for row in rows),
    }
    max_by_group = {key: value if value > 0.0 else 1.0 for key, value in max_by_group.items()}

    group_width = 1.0
    group_gap = 0.32
    group_starts = {
        group_key: index * (group_width + group_gap)
        for index, (group_key, _) in enumerate(groups)
    }
    x_min = -0.80
    x_max = group_starts[groups[-1][0]] + group_width + 0.48
    y_positions = [0.24 + index * 0.24 for index in reversed(range(len(rows)))]
    bar_height = 0.12
    label_font_size = AXIS_FONT_SIZE + 0.7
    header_y = y_positions[0] + 0.22
    row_label_x = -0.07

    fig, ax = plt.subplots(figsize=DEMAND_COVERAGE_FIGSIZE, dpi=FIGURE_DPI)
    ax.set_position([0.012, 0.012, 0.976, 0.976])
    ax.set_facecolor("white")

    # Column headers match the compact demand-coverage figure.
    for group_key, group_label in groups:
        ax.text(
            group_starts[group_key] + group_width / 2.0,
            header_y,
            f"{group_label}\n[GW]",
            ha="center",
            va="bottom",
            fontsize=label_font_size,
            color="#111827",
        )

    for y, row in zip(y_positions, rows):
        ax.text(
            row_label_x,
            y + 0.02,
            display_model_label(row["label"]),
            ha="right",
            va="center",
            fontsize=label_font_size - 0.1,
            color="#111827",
        )

        for group_key, _ in groups:
            start = group_starts[group_key]
            limit = float(row["limit"][group_key])
            limit_width = limit / max_by_group[group_key] * group_width

            # Grey background shows total capacity potential: finite expansion
            # limit plus fixed capacity without a finite p_nom_max.
            ax.barh(
                y,
                limit_width,
                left=start,
                height=bar_height,
                color=colors["Potential"],
                edgecolor="none",
                zorder=1,
            )

            if group_key == "pv":
                ongrid = float(row["installed"]["pv_ongrid"])
                offgrid = float(row["installed"]["pv_offgrid"])
                ongrid_width = ongrid / max_by_group[group_key] * group_width
                offgrid_width = offgrid / max_by_group[group_key] * group_width
                ax.barh(
                    y,
                    ongrid_width,
                    left=start,
                    height=bar_height,
                    color=colors["On-grid PV"],
                    edgecolor="none",
                    zorder=3,
                )
                ax.barh(
                    y,
                    offgrid_width,
                    left=start + ongrid_width,
                    height=bar_height,
                    color=colors["Off-grid PV"],
                    edgecolor="none",
                    zorder=3,
                )
                installed = ongrid + offgrid
                installed_width = ongrid_width + offgrid_width
            else:
                installed = float(row["installed"][group_key])
                installed_width = installed / max_by_group[group_key] * group_width
                ax.barh(
                    y,
                    installed_width,
                    left=start,
                    height=bar_height,
                    color=colors["Reservoir"] if group_key == "reservoir" else colors["ROR"],
                    edgecolor="none",
                    zorder=3,
                )

            # Label installed capacity and, if larger, the available capacity limit.
            ax.text(
                start + installed_width + 0.025,
                y,
                format_gw(installed),
                ha="left",
                va="center",
                fontsize=label_font_size - 0.4,
                color="#111827",
            )
            if limit > installed * 1.05 and limit_width - installed_width > 0.08:
                ax.text(
                    start + limit_width + 0.025,
                    y,
                    format_gw(limit),
                    ha="left",
                    va="center",
                    fontsize=label_font_size - 0.8,
                    color="#64748b",
                )

    legend_handles = [
        Patch(facecolor=colors["Potential"], edgecolor="white", label="Expansion limit"),
        Patch(facecolor=colors["Reservoir"], edgecolor="white", label="Reservoir"),
        Patch(facecolor=colors["ROR"], edgecolor="white", label="ROR"),
        Patch(facecolor=colors["On-grid PV"], edgecolor="white", label="On-grid PV"),
        Patch(facecolor=colors["Off-grid PV"], edgecolor="white", label="Off-grid PV"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=len(legend_handles),
        frameon=False,
        fontsize=label_font_size - 0.2,
        columnspacing=1.1,
        handlelength=1.0,
        handletextpad=0.35,
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.08, header_y + 0.20)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#111827")
        spine.set_linewidth(FRAME_LINE_WIDTH)
    return _save_figure(
        fig,
        "electricity_installed_capacity_by_carrier.png",
        fill_canvas=True,
        target_size_px=DEMAND_COVERAGE_IMAGE_SIZE_PX,
    )








# -----------------------------------------------------------
# Hydro Potentiale
# -----------------------------------------------------------


def plot_hydro_sites_potential_map(
    models: list[Model] | Model,
    filename: str = "hydropower_sites_potential_map.png",
    model_key: str | None = None,
) -> Path:
    """Plot modeled hydropower sites by project-size class and model-year availability.

    The map uses numbered markers instead of plant-name labels. Marker
    color indicates the model year in which the plant is available. Marker size
    indicates one of three project-size classes. Dense or overlapping marker
    numbers are moved outside the circles and connected to the plant location
    with a thin line.

    Parameters
    ----------
    models : list[Model] | Model
        Either one loaded model or the full model list.
    filename : str
        PNG filename written to ``results/elsevier``.
    model_key : str | None
        Optional model key used to select the country outline scenario.

    Returns
    -------
    Path
        Absolute path to the written PNG file.
    """
    import math
    import re
    import numpy as np

    # Accept either a single Model or the complete model list from load_models().
    if isinstance(models, list):
        if not models:
            raise ValueError("plot_hydro_sites_potential_map() received an empty model list.")
        model_list = models
    else:
        model_list = [models]

    # Select one model for static map resources such as the country outline.
    if model_key is None:
        map_model = model_list[0]
    else:
        matching_models = [candidate for candidate in model_list if candidate.key == model_key]
        if not matching_models:
            available = ", ".join(candidate.key for candidate in model_list)
            raise ValueError(
                f"No model with key '{model_key}' found. Available model keys: {available}"
            )
        map_model = matching_models[0]

    # Keep one representative model per model year.
    models_by_year: dict[int, Model] = {}
    for candidate in sorted(model_list, key=lambda item: item.year):
        models_by_year.setdefault(candidate.year, candidate)

    # Load only the country outline; regional boundaries are intentionally omitted.
    country_path = (
        PROJECT_ROOT
        / "resources"
        / map_model.scenario_name
        / "shapes"
        / "country_shapes.geojson"
    )
    if not country_path.exists():
        raise FileNotFoundError(f"Country outline not found: {country_path}")
    country = gpd.read_file(country_path).to_crs(epsg=4326)

    def normalise_column_label(value: Any) -> str:
        """Return a simplified column key for robust column matching."""
        return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())

    def normalise_site_key(value: Any) -> str:
        """Return a stable lowercase key for matching sites across model years."""
        text = str(value).strip().lower()
        text = re.sub(r"[^a-z0-9]+", "_", text)
        return text.strip("_")

    def pick_column(
        frame: pd.DataFrame,
        path: Path,
        candidates: list[str],
        required: bool = True,
    ) -> str | None:
        """Return the first existing column from a list of possible names."""
        lookup = {normalise_column_label(column): column for column in frame.columns}
        for candidate in candidates:
            key = normalise_column_label(candidate)
            if key in lookup:
                return lookup[key]
        if required:
            raise ValueError(f"{path.name} must contain one of: {', '.join(candidates)}")
        return None

    def find_powerplant_path(model: Model) -> Path:
        """Find the powerplants.csv file for one model scenario."""
        resource_dir = PROJECT_ROOT / "resources" / model.scenario_name
        candidates = [
            resource_dir / f"horizon_{model.year}" / "powerplants.csv",
            resource_dir / "powerplants.csv",
            resource_dir / "base_network" / "powerplants.csv",
            PROJECT_ROOT / "resources" / "powerplants.csv",
            PROJECT_ROOT / "data" / "powerplants.csv",
        ]
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if path is None:
            searched = "\n".join(str(candidate) for candidate in candidates)
            raise FileNotFoundError(f"No powerplants.csv found. Searched:\n{searched}")
        return path

    def display_technology(value: Any) -> str:
        """Convert technology tokens to compact paper-table labels."""
        text = str(value).strip()
        token = text.lower()
        if "run" in token or "ror" in token or "river" in token:
            return "Run-Of-River"
        if "reservoir" in token:
            return "Reservoir"
        if "phs" in token or "pumped" in token:
            return "Reservoir"
        return text if text else "--"

    def load_hydro_sites_from_path(
        powerplant_path: Path,
        default_year: int | None,
        source_rank: int,
    ) -> pd.DataFrame:
        """Load and clean hydropower sites from one powerplants.csv file."""
        powerplants = pd.read_csv(powerplant_path)

        # Identify required and optional columns robustly.
        lat_col = pick_column(powerplants, powerplant_path, ["lat", "latitude", "y"])
        lon_col = pick_column(powerplants, powerplant_path, ["lon", "longitude", "x"])
        name_col = pick_column(
            powerplants,
            powerplant_path,
            ["Name", "name", "Plant", "plant", "project", "Project"],
            required=False,
        )
        id_col = pick_column(
            powerplants,
            powerplant_path,
            ["id", "ID", "plant_id", "Plant_ID", "Name", "name"],
            required=False,
        )
        technology_col = pick_column(
            powerplants,
            powerplant_path,
            ["Technology", "technology", "Fueltype", "fueltype", "carrier", "Carrier"],
            required=False,
        )
        capacity_col = pick_column(
            powerplants,
            powerplant_path,
            [
                "Capacity",
                "capacity",
                "capacity_mw",
                "Potential",
                "potential",
                "potential_mw",
                "p_nom",
                "p_nom_max",
                "MW",
                "mw",
            ],
        )
        model_col = pick_column(
            powerplants,
            powerplant_path,
            ["Model", "model", "model_year", "Model Year", "year"],
            required=False,
        )

        # Keep only rows that are identifiable as hydropower technologies.
        hydro_mask = pd.Series(False, index=powerplants.index)
        for column in ["Fueltype", "fueltype", "Technology", "technology", "carrier", "Carrier"]:
            if column in powerplants.columns:
                hydro_mask = hydro_mask | powerplants[column].astype(str).str.lower().str.contains(
                    "hydro|river|reservoir|phs",
                    na=False,
                    regex=True,
                )

        hydro = powerplants[hydro_mask].copy()
        if hydro.empty:
            return pd.DataFrame()

        # Convert required plotting fields to internal columns.
        hydro["_lat"] = pd.to_numeric(hydro[lat_col], errors="coerce")
        hydro["_lon"] = pd.to_numeric(hydro[lon_col], errors="coerce")
        hydro["_capacity_mw"] = pd.to_numeric(hydro[capacity_col], errors="coerce")
        hydro["_name"] = (
            hydro[name_col].astype(str)
            if name_col is not None
            else hydro.index.to_series().astype(str)
        )
        hydro["_technology"] = (
            hydro[technology_col].map(display_technology)
            if technology_col is not None
            else "--"
        )

        # Use an explicit model-year column when available; otherwise use the scenario year.
        if model_col is not None:
            hydro["_model_year"] = pd.to_numeric(hydro[model_col], errors="coerce")
        else:
            hydro["_model_year"] = default_year

        # Build a site key for optional duplicate removal across scenario files.
        if id_col is not None:
            hydro["_site_key"] = hydro[id_col].map(normalise_site_key)
        else:
            hydro["_site_key"] = hydro["_name"].map(normalise_site_key)

        # Fall back to a coordinate-based key if a row has no usable name/id.
        missing_key = hydro["_site_key"].isin(["", "nan", "none"]) | hydro["_site_key"].isna()
        hydro.loc[missing_key, "_site_key"] = (
            hydro.loc[missing_key, "_lat"].round(4).astype(str)
            + "_"
            + hydro.loc[missing_key, "_lon"].round(4).astype(str)
        )

        # Preserve original table order within each source file.
        hydro["_source_order"] = [source_rank * 100000 + i for i in range(len(hydro))]

        # Drop incomplete or non-positive entries before plotting/exporting.
        hydro = hydro.dropna(subset=["_lat", "_lon", "_capacity_mw", "_model_year"])
        hydro = hydro[hydro["_capacity_mw"] > 0.0].copy()
        hydro["_model_year"] = hydro["_model_year"].astype(int)

        return hydro[
            [
                "_site_key",
                "_name",
                "_technology",
                "_capacity_mw",
                "_lat",
                "_lon",
                "_model_year",
                "_source_order",
            ]
        ]

    def find_dense_label_groups(
        ax: Any,
        sites: pd.DataFrame,
        marker_size_column: str,
    ) -> dict[int, dict[str, Any]]:
        """Return label positions, moving labels for sites that are visually too close."""
        sites = sites.reset_index(drop=True)
        n_sites = len(sites)

        # Default: every number is centered in its own circle.
        label_positions = {
            index: {
                "x": float(row["_lon"]),
                "y": float(row["_lat"]),
                "moved": False,
            }
            for index, row in sites.iterrows()
        }
        if n_sites <= 1:
            return label_positions

        # Use display coordinates because visual overlaps depend on rendered marker size.
        fig.canvas.draw()
        data_xy = sites[["_lon", "_lat"]].to_numpy(dtype=float)
        display_xy = ax.transData.transform(data_xy)
        marker_sizes = sites[marker_size_column].to_numpy(dtype=float)

        # Matplotlib scatter sizes are areas in pt²; convert approximate radius to pixels.
        marker_radius_px = np.sqrt(marker_sizes / math.pi) * fig.dpi / 72.0

        # Union-find groups all markers whose centered numbers would be too close.
        parent = list(range(n_sites))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left: int, right: int) -> None:
            root_left = find(left)
            root_right = find(right)
            if root_left != root_right:
                parent[root_right] = root_left

        # The threshold combines text proximity and circle overlap.
        for i in range(n_sites):
            for j in range(i + 1, n_sites):
                distance_px = float(np.linalg.norm(display_xy[i] - display_xy[j]))
                circle_threshold_px = 0.72 * (marker_radius_px[i] + marker_radius_px[j])
                text_threshold_px = 18.0
                if distance_px < max(text_threshold_px, circle_threshold_px):
                    union(i, j)

        components: dict[int, list[int]] = {}
        for index in range(n_sites):
            components.setdefault(find(index), []).append(index)

        inverse_transform = ax.transData.inverted()
        axis_box = ax.get_window_extent()
        margin_px = 8.0

        # Move labels for each dense component onto a surrounding ring.
        for component in components.values():
            if len(component) <= 1:
                continue

            component = sorted(component, key=lambda index: int(sites.iloc[index]["_no"]))
            component_points = display_xy[component]
            centroid = component_points.mean(axis=0)
            max_component_radius_px = max(
                float(np.linalg.norm(display_xy[index] - centroid) + marker_radius_px[index])
                for index in component
            )

            # Move every label only a short distance away from its own marker.
            start_angle = -math.pi / 2.0
            for order, index in enumerate(component):
                angle = start_angle + 2.0 * math.pi * order / len(component)
                label_radius_px = marker_radius_px[index] + max(3.0, 0.04 * len(component))
                label_display = display_xy[index] + label_radius_px * np.array(
                    [math.cos(angle), math.sin(angle)]
                )

                # Keep labels inside the visible map area whenever possible.
                label_display[0] = min(
                    max(label_display[0], axis_box.x0 + margin_px),
                    axis_box.x1 - margin_px,
                )
                label_display[1] = min(
                    max(label_display[1], axis_box.y0 + margin_px),
                    axis_box.y1 - margin_px,
                )

                label_data = inverse_transform.transform(label_display)
                label_positions[index] = {
                    "x": float(label_data[0]),
                    "y": float(label_data[1]),
                    "moved": True,
                }

        return label_positions

    # Prefer a single custom dataset if it already contains a Model column.
    map_powerplant_path = find_powerplant_path(map_model)
    map_powerplant_columns = pd.read_csv(map_powerplant_path, nrows=0).columns
    has_explicit_model_column = any(
        normalise_column_label(column) in {"model", "modelyear", "year"}
        for column in map_powerplant_columns
    )

    if has_explicit_model_column:
        hydro = load_hydro_sites_from_path(
            map_powerplant_path,
            default_year=None,
            source_rank=0,
        )
    else:
        # If no Model column exists, infer availability from the first scenario year in which the site appears.
        yearly_frames = []
        for source_rank, (year, year_model) in enumerate(sorted(models_by_year.items())):
            yearly_frames.append(
                load_hydro_sites_from_path(
                    find_powerplant_path(year_model),
                    default_year=year,
                    source_rank=source_rank,
                )
            )

        yearly_frames = [frame for frame in yearly_frames if not frame.empty]
        if not yearly_frames:
            raise ValueError("No hydropower plants found in any model year.")

        hydro = pd.concat(yearly_frames, axis=0, ignore_index=True)
        hydro = hydro.sort_values(["_site_key", "_model_year", "_source_order"])

        # Keep the earliest model year for each repeated site.
        hydro = hydro.drop_duplicates(subset=["_site_key"], keep="first")

    if hydro.empty:
        raise ValueError("No hydropower plants remain after filtering.")

    # Remove exact duplicate rows while keeping intentional same-name future projects.
    hydro = hydro.drop_duplicates(
        subset=["_name", "_technology", "_capacity_mw", "_lat", "_lon", "_model_year"],
        keep="first",
    )

    # Number plants in stable model-year and source-file order.
    hydro = hydro.sort_values(["_model_year", "_source_order", "_name"]).reset_index(drop=True)
    hydro["_no"] = range(1, len(hydro) + 1)

    def capacity_class(mw: float) -> str:
        """Map hydropower potential values to five globally comparable classes."""
        if mw < 10.0:
            return "0.5–10 MW"
        if mw < 50.0:
            return "10–50 MW"
        if mw < 500.0:
            return "50–500 MW"
        if mw < 2000.0:
            return "500–2,000 MW"
        return ">2,000 MW"

    # Use the same non-linear marker areas as the installed-capacity maps.
    # The smallest marker still contains the centered plant number.
    capacity_sizes = {
        "0.5–10 MW": 105,
        "10–50 MW": 200,
        "50–500 MW": 405,
        "500–2,000 MW": 760,
        ">2,000 MW": 1290,
    }
    capacity_order = list(capacity_sizes)
    technology_markers = {
        "Reservoir": "o",
        "Run-Of-River": "^",
    }

    # Add class labels and marker sizes to the hydro data.
    hydro["_capacity_class"] = hydro["_capacity_mw"].map(capacity_class)
    hydro["_marker_size"] = hydro["_capacity_class"].map(capacity_sizes)

    # Color sites by model-year availability with the original year colors.
    model_years = sorted(hydro["_model_year"].dropna().astype(int).unique())
    fallback_year_colors = {
        2025: "#1f4e79",
        2035: "#9c6644",
        2050: "#6a994e",
    }
    model_year_colors = {
        year: YEAR_COLORS.get(year, fallback_year_colors.get(year, plt.get_cmap("tab10")(i)))
        for i, year in enumerate(model_years)
    }

    # Compute map bounds from the country geometry and add a small visual margin.
    xmin, ymin, xmax, ymax = country.total_bounds
    pad_ratio = 0.035
    x_pad = max((xmax - xmin) * pad_ratio, 0.08)
    y_pad = max((ymax - ymin) * pad_ratio, 0.08)

    # Create the map canvas using the shared Elsevier export figure size.
    fig, ax = plt.subplots(figsize=TWO_COLUMN_EXPORT_FIGSIZE, dpi=FIGURE_DPI)
    ax.set_facecolor("white")

    # Draw the DRC outline as the only geographic background layer.
    country.boundary.plot(
        ax=ax,
        color="black",
        linewidth=COUNTRY_OUTLINE_LINE_WIDTH,
        zorder=1,
    )

    # Apply country bounds before overlap detection because label spacing uses display coordinates.
    ax.set_xlim(xmin - x_pad, xmax + x_pad)
    ax.set_ylim(ymin - y_pad, ymax + y_pad)
    ax.set_aspect("equal", adjustable="box")

    # Draw larger sites first so smaller sites and their numbers remain visible.
    plot_rows = hydro.sort_values("_marker_size", ascending=False)
    for _, row in plot_rows.iterrows():
        year = int(row["_model_year"])
        marker = technology_markers.get(row["_technology"], "o")
        ax.scatter(
            row["_lon"],
            row["_lat"],
            s=row["_marker_size"],
            marker=marker,
            facecolors=model_year_colors[year],
            edgecolors="#374151",
            linewidths=0.45,
            alpha=0.82,
            zorder=4,
        )

    # Determine which marker numbers have to be moved outside crowded circles.
    label_positions = find_dense_label_groups(
        ax=ax,
        sites=hydro,
        marker_size_column="_marker_size",
    )

    # Draw connector lines for labels that were moved outside their circles.
    for index, row in hydro.iterrows():
        label_position = label_positions[index]
        if not label_position["moved"]:
            continue

        ax.plot(
            [row["_lon"], label_position["x"]],
            [row["_lat"], label_position["y"]],
            color="#374151",
            linewidth=0.35,
            alpha=0.72,
            zorder=6,
        )

    # Write plant numbers. Centered labels remain in the circle; crowded labels are moved out.
    number_font_size = 6.2
    for index, row in hydro.iterrows():
        label_position = label_positions[index]
        label_color = "black" if label_position["moved"] else "white"
        label_x = label_position["x"]
        label_y = label_position["y"]

        # Move labels inside ROR triangles downward so they stay inside the marker.
        if not label_position["moved"] and row["_technology"] == "Run-Of-River":
            marker_radius_px = math.sqrt(float(row["_marker_size"]) / math.pi) * fig.dpi / 72.0
            label_display = ax.transData.transform([label_x, label_y])
            label_display[1] -= marker_radius_px * 0.34
            label_x, label_y = ax.transData.inverted().transform(label_display)

        ax.text(
            label_x,
            label_y,
            str(int(row["_no"])),
            fontsize=number_font_size,
            color=label_color,
            ha="center",
            va="center",
            zorder=7,
            clip_on=False,
        )

    # Build a separate legend for discrete hydropower project-size classes.
    capacity_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="#D8D8D8",
            markeredgecolor="#6b7280",
            markeredgewidth=0.35,
            markersize=math.sqrt(capacity_sizes[class_name]),
            label=class_name,
            alpha=0.82,
        )
        for class_name in capacity_order
    ]

    # Build a second legend for model-year availability.
    year_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=model_year_colors[year],
            markeredgecolor="#6b7280",
            markeredgewidth=0.35,
            markersize=7.0,
            label=str(year),
            alpha=0.82,
        )
        for year in model_years
    ]

    # Explain the marker shape independently from size and year color.
    technology_handles = [
        Line2D(
            [0],
            [0],
            marker=technology_markers[technology],
            linestyle="None",
            markerfacecolor="#D8D8D8",
            markeredgecolor="#374151",
            markeredgewidth=0.45,
            markersize=7.0,
            label=technology,
        )
        for technology in ["Reservoir", "Run-Of-River"]
        if (hydro["_technology"] == technology).any()
    ]

    # Place the capacity legend first and keep it as an additional artist.
    capacity_legend = ax.legend(
        handles=capacity_handles,
        title="Hydropower project size",
        loc="lower left",
        frameon=True,
        fontsize=AXIS_FONT_SIZE - 0.4,
        title_fontsize=AXIS_FONT_SIZE - 0.2,
        ncol=2,
        handlelength=2.8,
        handletextpad=1.0,
        columnspacing=1.2,
        labelspacing=1.7,
        handleheight=3.0,
        borderpad=0.9,
    )
    ax.add_artist(capacity_legend)

    # Place the model-year legend separately.
    year_legend = ax.legend(
        handles=year_handles,
        title="Model",
        loc="upper left",
        frameon=True,
        fontsize=AXIS_FONT_SIZE - 0.4,
        title_fontsize=AXIS_FONT_SIZE - 0.2,
        handletextpad=0.8,
        borderpad=0.6,
    )
    ax.add_artist(year_legend)

    # Place the technology legend separately.
    ax.legend(
        handles=technology_handles,
        title="Technology",
        loc="upper left",
        bbox_to_anchor=(0.0, 0.78),
        frameon=True,
        fontsize=AXIS_FONT_SIZE - 0.4,
        title_fontsize=AXIS_FONT_SIZE - 0.2,
        handletextpad=0.8,
        borderpad=0.6,
    )

    # Remove map axes.
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # Keep a simple black frame around the map for paper export.
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(FRAME_LINE_WIDTH)

    # Keep the figure itself untitled; use the caption in the paper instead.
    ax.set_title("")
    plt.tight_layout()

    # Save the PNG using the shared project export helper.
    return _save_figure(fig, filename)


def plot_installed_hydro_capacity_maps(
    models: list[Model],
    min_capacity_mw: float = 0.5,
) -> list[Path]:
    """Plot installed hydropower capacity for each solved result model.

    The function writes one PNG per model. Only hydropower components with an
    optimized capacity above ``min_capacity_mw`` are shown. Marker shape
    distinguishes reservoir and run-of-river plants; marker size uses five
    fixed classes for the optimized installed capacity.

    Parameters
    ----------
    models : list[Model]
        Loaded off-grid result models.
    min_capacity_mw : float
        Minimum optimized capacity in MW required for plotting a plant.

    Returns
    -------
    list[Path]
        Absolute paths to the written PNG files.
    """
    import math
    import re
    import numpy as np

    if not models:
        raise ValueError("plot_installed_hydro_capacity_maps() received an empty model list.")

    def site_index_from_component_name(component_name: Any) -> int | None:
        """Extract the powerplants.csv row index from a PyPSA component name."""
        match = re.match(r"^\s*(\d+)", str(component_name))
        return int(match.group(1)) if match else None

    def display_technology(value: Any) -> str:
        """Convert hydropower technology names to compact labels."""
        token = str(value).strip().lower()
        if "run" in token or "ror" in token or "river" in token:
            return "Run-Of-River"
        if "reservoir" in token or "hydro" in token:
            return "Reservoir"
        return "Reservoir"

    def capacity_class(mw: float) -> str:
        """Map optimized MW values to five globally comparable classes."""
        if mw < 10.0:
            return "0.5–10 MW"
        if mw < 50.0:
            return "10–50 MW"
        if mw < 500.0:
            return "50–500 MW"
        if mw < 2000.0:
            return "500–2,000 MW"
        return ">2,000 MW"

    def find_dense_label_groups(
        ax: Any,
        fig: plt.Figure,
        sites: pd.DataFrame,
        marker_size_column: str,
    ) -> dict[int, dict[str, Any]]:
        """Return label positions, moving labels for sites that are visually too close."""
        sites = sites.reset_index(drop=True)
        n_sites = len(sites)

        # Default: every number is centered in its own marker.
        label_positions = {
            index: {
                "x": float(row["_lon"]),
                "y": float(row["_lat"]),
                "moved": False,
            }
            for index, row in sites.iterrows()
        }
        if n_sites <= 1:
            return label_positions

        # Use display coordinates because visual overlaps depend on rendered marker size.
        fig.canvas.draw()
        data_xy = sites[["_lon", "_lat"]].to_numpy(dtype=float)
        display_xy = ax.transData.transform(data_xy)
        marker_sizes = sites[marker_size_column].to_numpy(dtype=float)

        # Matplotlib scatter sizes are areas in pt²; convert approximate radius to pixels.
        marker_radius_px = np.sqrt(marker_sizes / math.pi) * fig.dpi / 72.0

        # Union-find groups all markers whose centered numbers would be too close.
        parent = list(range(n_sites))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left: int, right: int) -> None:
            root_left = find(left)
            root_right = find(right)
            if root_left != root_right:
                parent[root_right] = root_left

        # The threshold combines text proximity and marker overlap.
        for i in range(n_sites):
            for j in range(i + 1, n_sites):
                distance_px = float(np.linalg.norm(display_xy[i] - display_xy[j]))
                marker_threshold_px = 0.72 * (marker_radius_px[i] + marker_radius_px[j])
                text_threshold_px = 18.0
                if distance_px < max(text_threshold_px, marker_threshold_px):
                    union(i, j)

        components: dict[int, list[int]] = {}
        for index in range(n_sites):
            components.setdefault(find(index), []).append(index)

        inverse_transform = ax.transData.inverted()
        axis_box = ax.get_window_extent()
        margin_px = 8.0

        # Move labels only a short distance away from their own marker.
        for component in components.values():
            if len(component) <= 1:
                continue

            component = sorted(component, key=lambda index: int(sites.iloc[index]["_no"]))
            start_angle = -math.pi / 2.0
            for order, index in enumerate(component):
                angle = start_angle + 2.0 * math.pi * order / len(component)
                label_radius_px = marker_radius_px[index] + max(3.0, 0.04 * len(component))
                label_display = display_xy[index] + label_radius_px * np.array(
                    [math.cos(angle), math.sin(angle)]
                )

                # Keep labels inside the visible map area whenever possible.
                label_display[0] = min(
                    max(label_display[0], axis_box.x0 + margin_px),
                    axis_box.x1 - margin_px,
                )
                label_display[1] = min(
                    max(label_display[1], axis_box.y0 + margin_px),
                    axis_box.y1 - margin_px,
                )

                label_data = inverse_transform.transform(label_display)
                label_positions[index] = {
                    "x": float(label_data[0]),
                    "y": float(label_data[1]),
                    "moved": True,
                }

        return label_positions

    # Use fixed non-linear classes so all five result maps are comparable.
    capacity_sizes = {
        "0.5–10 MW": 75,
        "10–50 MW": 145,
        "50–500 MW": 295,
        "500–2,000 MW": 560,
        ">2,000 MW": 960,
    }
    capacity_order = list(capacity_sizes)
    technology_markers = {
        "Reservoir": "o",
        "Run-Of-River": "^",
    }

    def installed_hydro_for_model(model: Model) -> tuple[Any, pd.DataFrame]:
        """Load installed hydropower rows and the country outline for one model."""
        network = model.network

        country_path = (
            PROJECT_ROOT
            / "resources"
            / model.scenario_name
            / "shapes"
            / "country_shapes.geojson"
        )
        if not country_path.exists():
            raise FileNotFoundError(f"Country outline not found: {country_path}")
        country = gpd.read_file(country_path).to_crs(epsg=4326)

        resource_dir = PROJECT_ROOT / "resources" / model.scenario_name
        powerplant_candidates = [
            resource_dir / f"horizon_{model.year}" / "powerplants.csv",
            resource_dir / "powerplants.csv",
        ]
        powerplant_path = next(
            (candidate for candidate in powerplant_candidates if candidate.exists()),
            None,
        )
        if powerplant_path is None:
            searched = "\n".join(str(candidate) for candidate in powerplant_candidates)
            raise FileNotFoundError(f"Powerplant table not found. Searched:\n{searched}")
        powerplants = pd.read_csv(powerplant_path)

        installed_rows = []

        # ROR hydropower is stored as PyPSA generators.
        generators = network.generators.copy()
        generator_carriers = generators["carrier"].astype(str).str.lower()
        hydro_generators = generators[
            generator_carriers.str.contains("ror|run|river", regex=True, na=False)
        ]
        for component_name, row in hydro_generators.iterrows():
            capacity_mw = row.get("p_nom_opt", row.get("p_nom", 0.0))
            capacity_mw = 0.0 if pd.isna(capacity_mw) else float(capacity_mw)
            if capacity_mw <= min_capacity_mw:
                continue

            site_index = site_index_from_component_name(component_name)
            if site_index is None or site_index not in powerplants.index:
                continue

            site = powerplants.loc[site_index]
            installed_rows.append(
                {
                    "_no": site_index + 1,
                    "_lat": float(site["lat"]),
                    "_lon": float(site["lon"]),
                    "_technology": display_technology(site.get("Technology", row.get("carrier"))),
                    "_capacity_mw": capacity_mw,
                    "_source_order": site_index,
                }
            )

        # Reservoir hydropower is stored as PyPSA storage units.
        storage_units = network.storage_units.copy()
        storage_carriers = storage_units["carrier"].astype(str).str.lower()
        hydro_storage = storage_units[
            storage_carriers.str.contains("hydro|reservoir", regex=True, na=False)
        ]
        for component_name, row in hydro_storage.iterrows():
            capacity_mw = row.get("p_nom_opt", row.get("p_nom", 0.0))
            capacity_mw = 0.0 if pd.isna(capacity_mw) else float(capacity_mw)
            if capacity_mw <= min_capacity_mw:
                continue

            site_index = site_index_from_component_name(component_name)
            if site_index is None or site_index not in powerplants.index:
                continue

            site = powerplants.loc[site_index]
            installed_rows.append(
                {
                    "_no": site_index + 1,
                    "_lat": float(site["lat"]),
                    "_lon": float(site["lon"]),
                    "_technology": display_technology(site.get("Technology", row.get("carrier"))),
                    "_capacity_mw": capacity_mw,
                    "_source_order": site_index,
                }
            )

        if not installed_rows:
            raise ValueError(f"No installed hydropower capacity found for model {model.key}.")

        hydro = pd.DataFrame(installed_rows).sort_values("_source_order").reset_index(drop=True)
        hydro["_capacity_class"] = hydro["_capacity_mw"].map(capacity_class)
        hydro["_marker_size"] = hydro["_capacity_class"].map(capacity_sizes)
        return country, hydro

    hydro_by_key = {
        model.key: installed_hydro_for_model(model)
        for model in models
    }

    def draw_installed_hydro_map(
        model: Model,
        country: Any,
        hydro: pd.DataFrame,
        fig: plt.Figure,
        ax: Any,
        show_legend: bool = True,
        legend_models: list[Model] | None = None,
        baseline_capacity_by_site: dict[int, float] | None = None,
        panel_label: str | None = None,
    ) -> None:
        """Draw one installed hydropower map on a supplied axis."""
        # Compute map bounds from the country geometry and add a small visual margin.
        xmin, ymin, xmax, ymax = country.total_bounds
        x_pad = max((xmax - xmin) * RESULT_MAP_PAD_RATIO, 0.02)
        y_pad = max((ymax - ymin) * RESULT_MAP_PAD_RATIO, 0.02)
        ax.set_facecolor("white")

        # Draw the DRC outline as the only geographic background layer.
        country.boundary.plot(
            ax=ax,
            color="black",
            linewidth=COUNTRY_OUTLINE_LINE_WIDTH,
            zorder=1,
        )

        # Apply country bounds before overlap detection because label spacing uses display coordinates.
        ax.set_xlim(xmin - x_pad, xmax + x_pad)
        ax.set_ylim(ymin - y_pad, ymax + y_pad)
        ax.set_aspect("equal", adjustable="box")

        # Draw larger sites first so smaller sites and their numbers remain visible.
        model_color = YEAR_COLORS.get(model.year, "#6b7280")
        baseline_capacity_by_site = baseline_capacity_by_site or {}
        expansion_tolerance_mw = 0.5
        plot_rows = hydro.sort_values("_marker_size", ascending=False)
        for _, row in plot_rows.iterrows():
            marker = technology_markers.get(row["_technology"], "o")
            site_number = int(row["_no"])
            baseline_capacity = baseline_capacity_by_site.get(site_number, 0.0)
            is_added_or_expanded = (
                model.year >= 2050
                and float(row["_capacity_mw"]) > baseline_capacity + expansion_tolerance_mw
            )
            marker_color = (
                YEAR_COLORS.get(2050, "#6a994e")
                if is_added_or_expanded
                else YEAR_COLORS.get(2035, "#9c6644")
                if model.year >= 2050 and baseline_capacity > 0.0
                else model_color
            )
            ax.scatter(
                row["_lon"],
                row["_lat"],
                s=row["_marker_size"],
                marker=marker,
                facecolors=marker_color,
                edgecolors="#374151",
                linewidths=0.45,
                alpha=0.82,
                zorder=4,
            )

        # Determine which marker numbers have to be moved outside crowded markers.
        label_positions = find_dense_label_groups(
            ax=ax,
            fig=fig,
            sites=hydro,
            marker_size_column="_marker_size",
        )

        # Draw connector lines for labels that were moved outside their markers.
        for index, row in hydro.iterrows():
            label_position = label_positions[index]
            if not label_position["moved"]:
                continue

            ax.plot(
                [row["_lon"], label_position["x"]],
                [row["_lat"], label_position["y"]],
                color="#374151",
                linewidth=0.35,
                alpha=0.72,
                zorder=6,
            )

        # Write plant numbers. Centered labels remain in the marker; crowded labels are moved out.
        number_font_size = 7.4
        for index, row in hydro.iterrows():
            label_position = label_positions[index]
            label_color = "black" if label_position["moved"] else "white"
            label_x = label_position["x"]
            label_y = label_position["y"]

            # Move labels inside ROR triangles downward so they stay inside the marker.
            if not label_position["moved"] and row["_technology"] == "Run-Of-River":
                marker_radius_px = math.sqrt(float(row["_marker_size"]) / math.pi) * fig.dpi / 72.0
                label_display = ax.transData.transform([label_x, label_y])
                label_display[1] -= marker_radius_px * 0.34
                label_x, label_y = ax.transData.inverted().transform(label_display)

            ax.text(
                label_x,
                label_y,
                str(int(row["_no"])),
                fontsize=number_font_size,
                color=label_color,
                ha="center",
                va="center",
                zorder=7,
                clip_on=False,
            )

        if panel_label is not None:
            ax.text(
                0.98,
                0.97,
                panel_label,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=LEGEND_FONT_SIZE,
                color="#111827",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "#d1d5db",
                    "boxstyle": "round,pad=0.22",
                    "alpha": 0.92,
                },
                zorder=10,
            )

        if not show_legend:
            # Remove map axes and finish this panel without legends.
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("black")
                spine.set_linewidth(FRAME_LINE_WIDTH)
            ax.set_title("")
            return

        # Build a legend for optimized installed capacity classes.
        capacity_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markerfacecolor="#D8D8D8",
                markeredgecolor="#6b7280",
                markeredgewidth=0.35,
                markersize=math.sqrt(capacity_sizes[class_name]) * 1.04,
                label=class_name,
                alpha=0.82,
            )
            for class_name in capacity_order
        ]

        if baseline_capacity_by_site and model.year >= 2050:
            model_handle = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="None",
                    markerfacecolor=YEAR_COLORS.get(2035, "#9c6644"),
                    markeredgecolor="#6b7280",
                    markeredgewidth=0.35,
                    markersize=8.6,
                    label="2035 capacity",
                    alpha=0.82,
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="None",
                    markerfacecolor=YEAR_COLORS.get(2050, "#6a994e"),
                    markeredgecolor="#6b7280",
                    markeredgewidth=0.35,
                    markersize=8.6,
                    label="Added or expanded by 2050",
                    alpha=0.82,
                ),
            ]
            model_legend_title = "Site status"
        else:
            # Build a compact model legend for the current result scenario.
            legend_models = legend_models if legend_models is not None else [model]
            model_handle = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="None",
                    markerfacecolor=YEAR_COLORS.get(legend_model.year, "#6b7280"),
                    markeredgecolor="#6b7280",
                    markeredgewidth=0.35,
                    markersize=8.6,
                    label=legend_model.label,
                    alpha=0.82,
                )
                for legend_model in legend_models
            ]
            model_legend_title = "Model"

        # Explain the marker shape independently from size and model color.
        technology_handles = [
            Line2D(
                [0],
                [0],
                marker=technology_markers[technology],
                linestyle="None",
                markerfacecolor="#D8D8D8",
                markeredgecolor="#374151",
                markeredgewidth=0.45,
                markersize=8.6,
                label=technology,
            )
            for technology in ["Reservoir", "Run-Of-River"]
            if (hydro["_technology"] == technology).any()
        ]

        # Place the capacity legend first and keep it as an additional artist.
        capacity_legend = ax.legend(
            handles=capacity_handles,
            title="Installed hydropower capacity",
            loc="lower left",
            frameon=True,
            fontsize=AXIS_FONT_SIZE + 1.0,
            title_fontsize=AXIS_FONT_SIZE + 1.2,
            ncol=2,
            handlelength=2.8,
            handletextpad=1.0,
            columnspacing=1.2,
            labelspacing=1.7,
            handleheight=3.0,
            borderpad=0.9,
        )
        ax.add_artist(capacity_legend)

        # Place the model legend separately.
        model_legend = ax.legend(
            handles=model_handle,
            title=model_legend_title,
            loc="upper left",
            frameon=True,
            fontsize=AXIS_FONT_SIZE + 1.0,
            title_fontsize=AXIS_FONT_SIZE + 1.2,
            handletextpad=0.8,
            borderpad=0.6,
        )
        ax.add_artist(model_legend)

        # Place the technology legend below the model legend.
        ax.legend(
            handles=technology_handles,
            title="Technology",
            loc="upper left",
            bbox_to_anchor=(0.0, 0.84),
            frameon=True,
            fontsize=AXIS_FONT_SIZE + 1.0,
            title_fontsize=AXIS_FONT_SIZE + 1.2,
            handletextpad=0.8,
            borderpad=0.6,
        )

        # Remove map axes.
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        # Keep a simple black frame around the map for paper export.
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(FRAME_LINE_WIDTH)

        # Keep the figure itself untitled; use the caption in the paper instead.
        ax.set_title("")

    output_paths: list[Path] = []
    for model in models:
        country, hydro = hydro_by_key[model.key]
        fig, ax = plt.subplots(figsize=TWO_COLUMN_EXPORT_FIGSIZE, dpi=FIGURE_DPI)
        ax.set_position([0.0, 0.0, 1.0, 1.0])
        draw_installed_hydro_map(
            model=model,
            country=country,
            hydro=hydro,
            fig=fig,
            ax=ax,
            show_legend=True,
        )

        output_paths.append(
            _save_figure(
                fig,
                f"hydropower_installed_capacity_map_{model.key}.png",
                fill_canvas=True,
            )
        )

    models_by_key = {model.key: model for model in models}

    combined_keys = ("myopic_2035_0", "myopic_2035_early_large")
    if all(key in models_by_key for key in combined_keys):
        combined_models = [models_by_key[key] for key in combined_keys]
        baseline_hydro = hydro_by_key.get("myopic_2035_0")
        baseline_capacity_by_site = (
            baseline_hydro[1].set_index("_no")["_capacity_mw"].astype(float).to_dict()
            if baseline_hydro is not None
            else {}
        )
        combined_fig, axes = plt.subplots(
            len(combined_models),
            1,
            figsize=(
                TWO_COLUMN_EXPORT_FIGSIZE[0],
                TWO_COLUMN_EXPORT_FIGSIZE[1] * len(combined_models),
            ),
            dpi=FIGURE_DPI,
        )
        panel_height = 1.0 / len(combined_models)
        for index, axis in enumerate(axes):
            axis.set_position(
                [0.0, 1.0 - (index + 1) * panel_height, 1.0, panel_height]
            )

        for index, (axis, model) in enumerate(zip(axes, combined_models)):
            country, hydro = hydro_by_key[model.key]
            draw_installed_hydro_map(
                model=model,
                country=country,
                hydro=hydro,
                fig=combined_fig,
                ax=axis,
                show_legend=(index == len(combined_models) - 1),
                legend_models=combined_models,
                baseline_capacity_by_site=baseline_capacity_by_site,
                panel_label=model.label,
            )

        output_paths.append(
            _save_figure(
                combined_fig,
                "hydropower_installed_capacity_map_myopic_2035_0_2035_early_large_combined.png",
                fill_canvas=True,
                target_size_px=(
                    TWO_COLUMN_IMAGE_SIZE_PX[0],
                    TWO_COLUMN_IMAGE_SIZE_PX[1] * len(combined_models),
                ),
            )
        )

    combined_keys = ("myopic_2050_no_large_hydro", "myopic_2050_0")
    if all(key in models_by_key for key in combined_keys):
        combined_models = [models_by_key[key] for key in combined_keys]
        baseline_hydro = hydro_by_key.get("myopic_2035_0")
        baseline_capacity_by_site = (
            baseline_hydro[1].set_index("_no")["_capacity_mw"].astype(float).to_dict()
            if baseline_hydro is not None
            else {}
        )
        combined_fig, axes = plt.subplots(
            len(combined_models),
            1,
            figsize=(
                TWO_COLUMN_EXPORT_FIGSIZE[0],
                TWO_COLUMN_EXPORT_FIGSIZE[1] * len(combined_models),
            ),
            dpi=FIGURE_DPI,
        )
        panel_height = 1.0 / len(combined_models)
        for index, axis in enumerate(axes):
            axis.set_position(
                [0.0, 1.0 - (index + 1) * panel_height, 1.0, panel_height]
            )

        for index, (axis, model) in enumerate(zip(axes, combined_models)):
            country, hydro = hydro_by_key[model.key]
            draw_installed_hydro_map(
                model=model,
                country=country,
                hydro=hydro,
                fig=combined_fig,
                ax=axis,
                show_legend=(index == len(combined_models) - 1),
                legend_models=combined_models,
                baseline_capacity_by_site=baseline_capacity_by_site,
                panel_label=model.label,
            )

        output_paths.append(
            _save_figure(
                combined_fig,
                "hydropower_installed_capacity_map_myopic_2035_2050_no_large_2050_0_combined.png",
                fill_canvas=True,
                target_size_px=(
                    TWO_COLUMN_IMAGE_SIZE_PX[0],
                    TWO_COLUMN_IMAGE_SIZE_PX[1] * len(combined_models),
                ),
            )
        )

    return output_paths











# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Load myopic models and create quick-look result plots.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    # Load all networks once and reuse them for every plot.
    models = load_models()

    # Export one result-network map for each model.
    for model in models:
        plot_result_network_topology(model)
    plot_h2_pipeline_flow_maps(models)
    plot_electricity_demand_node_maps(models)

    # Export the separated electricity demand-coverage stack.
    plot_electricity_demand_coverage_stack(models)
    plot_system_costs_capex_opex_stack(models)
    plot_dual_price_lcoh_comparison(models)
    plot_installed_capacity_by_carrier(models)

    plot_hydro_sites_potential_map(models, model_key="myopic_2050_133p32")
    plot_installed_hydro_capacity_maps(models)

if __name__ == "__main__":
    main()
