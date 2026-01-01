# src/TRITON_SWMM_toolkit/config_model.py
from pydantic import BaseModel, Field, field_validator
from pathlib import Path
import yaml
import numpy as np


class TS_config(BaseModel):
    # FILEPATHS
    variable_boundary_condition: Path = Field(
        "n/a",
        description="Path to shapefile representing extent of variable water level boundary condition.",
    )
    mannings_file: Path = Field(
        "n/a",
        description="manning's roughness filepath (required if a constant mannings is not defined)",
    )
    watershed_shapefile: Path = Field("n/a", description="Directory to store outputs")
    DEM_processed: Path = Field("n/a", description="DEM formatted for TRITON")
    DEM_fullres: Path = Field(
        "n/a", description="DEM to be formatted and, if desired, coarsened, for TRITON"
    )
    landuse_lookup_file: Path = Field(
        "n/a",
        description="CSV file containing lookup table relating landuse categories to manning's roughness coefficients",
    )
    SWMM_hydraulics: Path = Field(
        "n/a",
        description="Hydraulics-only SWMM model template with fillable fields based on input weather data. An event-specific scenario of this model will be input to TRITON-SWMM.",
    )
    SWMM_hydrology: Path = Field(
        "n/a",
        description="Hydrology-only SWMM model template with fillable fields based on input weather data. This will be run prior to TRITON-SWMM to generate runoff time series in grid cells that overlap with subcatchment outlet nodes.",
    )
    SWMM_full: Path = Field(
        "n/a",
        description="Full SWMM model template with fillable fields based on input weather data. Scenarios based on this can be run in addition to TRITON-SWMM to compare SWMM hydraulics results.",
    )
    landuse_raster: Path = Field(
        "n/a",
        description="Landuse raster used for creating manning's roughness input.",
    )
    TRITONSWMM_software_directory: Path = Field(
        "n/a",
        description="Folder containing the TRITON-SWMM model version used for a particular simulation.",
    )
    weather_timeseries: Path = Field(
        "n/a",
        description="Netcdf containing weather event time series data.",
    )
    weather_event_summary_csv: Path = Field(
        "n/a",
        description="CSV file with weather event summary statistics.",
    )
    subcatchment_raingage_mapping: Path = Field(
        "n/a",
        description="Lookup table relating spatially indexed rainfall time series to SWMM subcatchment IDs.",
    )
    benchmarking_experiment: Path = Field(
        "n/a",
        description="Benchmarking experimental design.",
    )
    # ATTRIBUTES
    landuse_description_colname: str = Field(
        "original_description",
        description="column name in the landuse_lookup_file corresponding to landuse description.",
    )
    landuse_lookup_class_id_colname: str = Field(
        "CLASS_ID",
        description="column name in the landuse_lookup_file corresponding to landuse classification.",
    )
    landuse_lookup_mannings_colname: str = Field(
        "mannings",
        description="column name in the landuse_lookup_file corresponding to manning's coefficient.",
    )
    landuse_plot_color_colname: str = Field(
        "plot_color",
        description="column name in the landuse_lookup_file corresponding to target plot colors by landuse.",
    )
    # CONSTANTS
    dem_outside_watershed_height: float = Field(
        np.nan,
        description="DEM height applied to grid cells outside of the watershed boundary.",
    )
    dem_building_height: float = Field(
        np.nan,
        description="DEM height applied to DEM gridcells overlapping buildings.",
    )
    # PARAMETERS
    target_dem_resolution: float = Field(
        np.nan,
        description="Target DEM resolution for TRITON-SWMM in the native resolution of the provided DEM.",
    )


def load_toolkit_config(cfg):
    toolkit = TS_config.model_validate(cfg)
    return toolkit
