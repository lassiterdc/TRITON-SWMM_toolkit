# src/TRITON_SWMM_toolkit/config_model.py
from pydantic import BaseModel, Field, field_validator
from pathlib import Path
import yaml


class TS_config(BaseModel):
    # needed for running simulations
    DEM: Path = Field("n/a", description="DEM formatted for TRITON")
    variable_boundary_condition: Path = Field(
        "n/a",
        description="Path to shapefile representing extent of variable water level boundary condition.",
    )
    mannings_file: Path = Field(
        "n/a",
        description="manning's roughness filepath (required if a constant mannings is not defined)",
    )
    # needed for plotting
    watershed_shapefile: Path = Field("n/a", description="Directory to store outputs")


def load_toolkit_config(cfg):
    toolkit = TS_config.model_validate(cfg)
    return toolkit
