"""Console script for TRITON_SWMM_toolkit."""

import typer
from rich.console import Console
from TRITON_SWMM_toolkit import run_model

app = typer.Typer()
console = Console()


@app.command()
def main(
    config: str = typer.Option(..., help="Path to config file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Run TRITON-SWMM in terminal (TUI)."""

    console.print(f"[bold green]Running TRITON-SWMM with config:[/bold green] {config}")
    run_model(config_path=config, verbose=verbose)
    console.print("[bold blue]Simulation finished![/bold blue]")


if __name__ == "__main__":
    app()
