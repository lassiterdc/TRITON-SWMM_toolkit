# src/TRITON_SWMM_toolkit/config_model.py
from pydantic import BaseModel, Field, field_validator
from pathlib import Path
import yaml


class SimulationConfig(BaseModel):
    input_file: Path = Field(..., description="Path to SWMM input file")
    output_dir: Path = Field(..., description="Directory to store outputs")
    timestep: int = Field(5, ge=1, description="Simulation timestep in minutes")
    verbose: bool = Field(False, description="Enable verbose logging")

    # Validate input_file exists
    @field_validator("input_file")
    def check_input_file_exists(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"Input file does not exist: {v}")
        return v

    # Ensure output_dir exists
    @field_validator("output_dir")
    def create_output_dir(cls, v: Path) -> Path:
        v.mkdir(parents=True, exist_ok=True)
        return v

    # Load from YAML
    @classmethod
    def from_yaml(cls, yaml_path: Path):
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)
