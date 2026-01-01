import numpy as np
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import colormaps
import pandas as pd
import rioxarray as rxr
import geopandas as gpd
from pathlib import Path
import xarray as xr
from typing import Optional
from matplotlib.axes import Axes
from TRITON_SWMM_toolkit.prepare_simulations import create_mannings_raster


def plot_polygon_boundary_on_ax(ax, shp_path: Path, color="black", linewidth=1):
    gdf = gpd.read_file(shp_path)
    gdf.boundary.plot(ax=ax, color=color, linewidth=linewidth)


def plot_discrete_raster(
    rds,
    cbar_lab,
    cmap: str = "",  # requires either this or colors
    colors: pd.Series = pd.Series(),  # colors indexed by value
    labs: pd.Series = pd.Series(),  # description for tick labels on colorbar indexed by value
    label_axes_ticks=False,  # determines whether or not x and y axis ticks are labeled
    ax: Optional[Axes] = None,
    watershed_shapefile: Path = Path(""),
    watershed_shapefile_color="black",
):
    # filter out no data values
    nodata = rds.rio.nodata
    if nodata is not None:
        mask = rds.values != nodata
    else:
        mask = ~np.isnan(rds.values)
    rds = rds.where(mask)
    unique_vals = np.unique(rds.values[~np.isnan(rds.values)])
    unique_vals.sort()
    if len(colors) > 0:
        cs = colors.loc[unique_vals]
        cmap_obj = ListedColormap(cs)
    else:
        cmap_obj = colormaps[cmap]

    bounds = np.append(unique_vals, unique_vals[-1] + 1)
    norm = mcolors.BoundaryNorm(bounds, cmap_obj.N)
    tick_positions = (bounds[:-1] + bounds[1:]) / 2
    if ax is None:
        fig, ax = plt.subplots()
    ax.set_aspect("equal", adjustable="box")

    img = rds.plot(ax=ax, cmap=cmap_obj, norm=norm, add_colorbar=False)

    cbar = fig.colorbar(img, ax=ax, boundaries=bounds, ticks=tick_positions)

    if len(labs) > 0:
        ticklabels = labs.loc[unique_vals]
    else:
        ticklabels = [str((val)) for val in unique_vals]
    cbar.ax.set_yticklabels(ticklabels)
    cbar.set_label(cbar_lab)
    if not label_axes_ticks:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlabel("")
        ax.set_ylabel("")
    ax.set_title("")
    if len(watershed_shapefile.name) > 0:
        plot_polygon_boundary_on_ax(
            ax, watershed_shapefile, color=watershed_shapefile_color
        )
    # plt.show()

    return ax


def plot_landuse_raster(
    landuse_raster,
    landuse_lookup,
    landuse_colname,
    landuse_description_colname,
    landuse_plot_color_colname,
    watershed_shapefile,
    watershed_shapefile_color="black",
    ax: Optional[Axes] = None,
):
    rds = rxr.open_rasterio(landuse_raster)
    df_lu_lookup = pd.read_csv(landuse_lookup).set_index(landuse_colname)
    labs = df_lu_lookup[landuse_description_colname]
    colors = df_lu_lookup[landuse_plot_color_colname]

    ax = plot_discrete_raster(
        rds,
        cbar_lab="",
        colors=colors,
        labs=labs,
        watershed_shapefile=watershed_shapefile,
        watershed_shapefile_color=watershed_shapefile_color,
        ax=ax,
    )
    return ax


def plot_continuous_raster(
    rds: xr.DataArray,
    cbar_lab: str,
    cmap: str = "viridis",
    label_axes_ticks: bool = False,
    watershed_shapefile: Optional[Path] = None,
    watershed_shapefile_color: str = "black",
    ax: Optional[Axes] = None,
    show_cbar: bool = True,
    vmin=None,
    vmax=None,
    **cbar_kwargs,
):
    # Mask nodata values if present
    nodata = rds.rio.nodata if hasattr(rds, "rio") else None
    mask = ~np.isnan(rds.values) if nodata is None else (rds.values != nodata)
    rds = rds.where(mask)

    if ax is None:
        fig, ax = plt.subplots()
    ax.set_aspect("equal", adjustable="box")
    if vmin is None:
        vmin = rds.min()

    if vmax is None:
        vmax = rds.max()

    cmap_obj = colormaps[cmap].copy()
    for key in ["set_under", "set_over", "set_bad"]:
        if key in cbar_kwargs:
            getattr(cmap_obj, key)(cbar_kwargs[key])

    img = rds.plot(  # type: ignore
        ax=ax,
        cmap=cmap_obj,
        add_colorbar=show_cbar,
        vmin=vmin,
        vmax=vmax,
    )
    if show_cbar:
        img.colorbar.set_label(cbar_lab)

    if not label_axes_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")

    ax.set_title("")

    if watershed_shapefile and watershed_shapefile.exists():
        plot_polygon_boundary_on_ax(
            ax,
            watershed_shapefile,
            color=watershed_shapefile_color,
        )

    # plt.show()
    return ax


def process_dem_for_plotting(rds_dem, dem_out_of_watershed, dem_building_height):
    """
    Docstring for process_dem_for_plotting

    :param rds_dem: dem dataset
    :param dem_out_of_watershed: height assigned to DEM outside of watershed
    :param dem_building_height: height assigned to DEM where it overlaps buildings

    This assigns n/a to all dem grid cells that are outside of the watershed or represent buildings
    """
    rds_dem_plot_ready = rds_dem.where(
        (rds_dem != dem_out_of_watershed) & (rds_dem != dem_building_height)
    )
    return rds_dem_plot_ready


def plot_fullres_vs_coarse_dem(
    dem_outside_watershed_height,
    dem_building_height,
    dem_unprocessed,
    dem_processed,
    watershed_shapefile,
    vmin=None,
    vmax=None,
):
    rds_dem_unprocessed = rxr.open_rasterio(dem_unprocessed)
    rds_dem_processed = rxr.open_rasterio(dem_processed)

    rds_dem_fullres_for_plotting = process_dem_for_plotting(
        rds_dem_unprocessed, dem_outside_watershed_height, dem_building_height
    )

    if vmin is None:
        vmin = rds_dem_fullres_for_plotting.min()
    if vmax is None:
        vmax = rds_dem_fullres_for_plotting.max()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    ax1 = plot_continuous_raster(
        rds_dem_unprocessed,  # type: ignore
        cbar_lab="elevation",
        cmap="terrain",
        watershed_shapefile=watershed_shapefile,
        watershed_shapefile_color="red",
        vmin=vmin,
        vmax=vmax,
        ax=axes[0],
        set_over="white",
        show_cbar=False,
    )
    ax1.set_title("full res DEM")
    ax2 = plot_continuous_raster(
        rds_dem_processed,  # type: ignore
        cbar_lab="elevation",
        cmap="terrain",
        watershed_shapefile=watershed_shapefile,
        watershed_shapefile_color="red",
        vmin=vmin,
        vmax=vmax,
        ax=axes[1],
        set_over="white",
    )
    ax2.set_title("coarsened DEM")
    return axes


def plot_fullres_vs_coarse_mannings(
    mannings_processed,
    landuse_lookup,
    landuse_raster,
    landuse_colname,
    mannings_colname,
    watershed_shapefile,
):
    rds_mannings = create_mannings_raster(
        landuse_lookup, landuse_raster, landuse_colname, mannings_colname
    )
    rds_mannings_processed = rxr.open_rasterio(mannings_processed)

    vmin = rds_mannings.min()
    vmax = rds_mannings.max()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    ax1 = plot_continuous_raster(
        rds_mannings,
        cbar_lab="mannings",
        cmap="viridis",
        watershed_shapefile=watershed_shapefile,
        watershed_shapefile_color="red",
        vmin=vmin,
        vmax=vmax,
        ax=axes[0],
        show_cbar=False,
    )
    ax1.set_title("full res mannings")
    ax2 = plot_continuous_raster(
        rds_mannings_processed,  # type: ignore
        cbar_lab="mannings",
        cmap="viridis",
        watershed_shapefile=watershed_shapefile,
        watershed_shapefile_color="red",
        vmin=vmin,
        vmax=vmax,
        ax=axes[1],
    )
    ax2.set_title("coarsened res mannings")
    return axes
