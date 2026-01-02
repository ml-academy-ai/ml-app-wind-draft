"""Programmatic entrypoint for running the Kedro training pipeline."""

import os
import sys
import tomllib
from pathlib import Path

from kedro.framework.project import configure_project
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# Change to project directory so relative paths resolve correctly
os.chdir(project_root)


def run_training_pipeline(
    env: str = "local",
    pipeline_name: str = "training",
) -> None:
    """Run the Kedro training pipeline programmatically."""
    # Extract package name from pyproject.toml
    with open(project_root / "pyproject.toml", "rb") as f:
        package_name = tomllib.load(f)["tool"]["kedro"]["package_name"]

    configure_project(package_name)
    bootstrap_project(project_root)

    with KedroSession.create(project_path=project_root, env=env) as session:
        session.run(pipeline_name=pipeline_name)


if __name__ == "__main__":
    run_training_pipeline()
