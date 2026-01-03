# %%
import requests
from pathlib import Path
from TRITON_SWMM_toolkit.constants import (
    APP_NAME,
    NORFOLK_EX,
    DOWNLOAD_EXAMPLES_IF_ALREADY_EXIST,
    NORFOLK_SYSTEM_CONFIG,
    NORFOLK_SINGLE_SIM_EXP,
    NORFOLK_BENCHMARKING_EXP,
    NORFOLK_CASE_CONFIG,
)
from platformdirs import user_data_dir
from importlib.resources import files
import yaml
from zipfile import ZipFile
import bagit
import shutil
from TRITON_SWMM_toolkit.config import (
    load_system_config,
    load_benchmarking_experiment_config_config,
)

try:
    from hsclient import HydroShare
except ImportError:
    HydroShare = None


def get_data_root() -> Path:
    return Path(user_data_dir(APP_NAME))


def load_config_file(case_study_name: str, filename: str) -> dict:
    path = files(APP_NAME).joinpath(f"examples/{case_study_name}/{filename}")
    return yaml.safe_load(path.read_text())


def download_data_from_hydroshare(
    res_identifier: str,
    target: Path,
    hs,
    download_if_exists=DOWNLOAD_EXAMPLES_IF_ALREADY_EXIST,
    validate=False,
):
    if target.exists() and download_if_exists:
        shutil.rmtree(target)
    if target.exists() and not download_if_exists:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    hs_resource = hs.resource(res_identifier)
    zip_path = Path(hs_resource.download(target.parent))
    extract_to = target.parent

    with ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)

    # Detect the actual top-level folder
    with ZipFile(zip_path, "r") as z:
        top_level_dirs = {Path(f).parts[0] for f in z.namelist() if Path(f).parts}
        if len(top_level_dirs) == 1:
            unzipped_folder = extract_to / next(iter(top_level_dirs))
        else:
            raise RuntimeError(
                "ZIP has multiple top-level folders; cannot determine Bag root."
            )
    # unzipped_folder = Path(extract_to).joinpath(zip_path.name.split(".")[0])
    if validate:
        bag = bagit.Bag(unzipped_folder)
        if bag.is_valid():
            print("Bag verified! All checksums match.")
        else:
            print("Bag is invalid!")

    outdir = unzipped_folder.rename(target)
    zip_path.unlink()


def sign_into_hydroshare():
    if HydroShare is None:
        raise RuntimeError(
            "hsclient is not installed. Install optional dependencies with `pip install .[tests]`. Alternatively, you can download the data manually if you have issues installing this package with pip. Link: https://www.hydroshare.org/resource/a4aace329b8c401a93e94ce2a761fe1b/"
        )
    hs = HydroShare()
    hs.sign_in()
    print("signed into Hydroshare successfully.")
    return hs


download_if_exists = DOWNLOAD_EXAMPLES_IF_ALREADY_EXIST


def load_norfolk_system_config(
    download_if_exists=DOWNLOAD_EXAMPLES_IF_ALREADY_EXIST,
):
    case_details = load_config_file(NORFOLK_EX, NORFOLK_CASE_CONFIG)
    res_identifier = case_details["res_identifier"]  # will come from the case yaml
    target = get_data_root() / "examples" / NORFOLK_EX
    data_dir = target / "data" / "contents"
    cfg_template = load_config_file(NORFOLK_EX, NORFOLK_SYSTEM_CONFIG)
    cfg_filled = {
        key: (
            value.format(DATA_DIR=str(data_dir), DATA=str(data_dir))
            if isinstance(value, str)
            else value
        )
        for key, value in cfg_template.items()
    }
    model = load_system_config(cfg_filled)
    if target.exists() and not download_if_exists:
        pass
    else:
        hs = sign_into_hydroshare()
        download_data_from_hydroshare(
            res_identifier, target, hs, download_if_exists=download_if_exists
        )

        zipped_software = Path(str(model.TRITONSWMM_software_directory) + ".zip")

        with ZipFile(zipped_software, "r") as z:
            z.extractall(model.TRITONSWMM_software_directory.parent)

        zipped_software.unlink()

    return model


def load_experiment_config(experiment_config_filename: str):
    case_details = load_config_file(NORFOLK_EX, NORFOLK_CASE_CONFIG)
    target = get_data_root() / "examples" / NORFOLK_EX
    data_dir = target / "data" / "contents"
    cfg_template = load_config_file(NORFOLK_EX, experiment_config_filename)
    cfg_filled = {
        key: (
            value.format(DATA_DIR=str(data_dir), DATA=str(data_dir))
            if isinstance(value, str)
            else value
        )
        for key, value in cfg_template.items()
    }
    return load_benchmarking_experiment_config_config(cfg_filled)


def load_norfolk_benchmarking_config():
    return load_experiment_config(NORFOLK_BENCHMARKING_EXP)


def load_norfolk_single_sim_experiment():
    return load_experiment_config(NORFOLK_SINGLE_SIM_EXP)
