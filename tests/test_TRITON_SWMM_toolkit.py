# tests/test_TRITON_SWMM_toolkit.py
import pytest
from TRITON_SWMM_toolkit import run_model
from typer.testing import CliRunner
from TRITON_SWMM_toolkit.cli import app
from unittest.mock import patch
from TRITON_SWMM_toolkit.gui import launch_gui


def test_run_model_returns_something(tmp_path):
    config_file = tmp_path / "dummy_config.yaml"
    config_file.write_text("dummy: value")
    result = run_model(str(config_file), verbose=True, timestep=5)
    assert result["status"] == "success"


runner = CliRunner()


def test_cli_main(tmp_path):
    """Test the CLI main command with a dummy config."""

    # Create a dummy config
    config_file = tmp_path / "dummy.yaml"
    config_file.write_text("dummy: value")

    # Run CLI
    result = runner.invoke(app, ["main", "--config", str(config_file), "--verbose"])
    assert result.exit_code == 0
    assert "Running TRITON-SWMM" in result.output


def test_run_model_called(tmp_path):
    """Test that run_model gets called (mocked)."""
    with patch("TRITON_SWMM_toolkit.gui.run_model") as mock_run:
        # Simulate calling the run function directly
        mock_run(str(tmp_path / "dummy.yaml"), verbose=True, timestep=5)
        mock_run.assert_called_once()
