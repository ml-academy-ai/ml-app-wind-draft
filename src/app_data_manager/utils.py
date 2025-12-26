from pathlib import Path

import yaml


def read_config(config_path: str | Path) -> dict:
    """
    Reads a YAML configuration file and returns it as a dictionary.

    Parameters:
    ----------
    config_path : str or Path
        Path to the YAML file.

    Returns:
    -------
    dict
        Parsed YAML content as a dictionary.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r") as f:
        return yaml.safe_load(f)
