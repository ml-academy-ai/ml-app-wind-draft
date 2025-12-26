"""Programmatic entrypoint for running the Kedro inference pipeline."""

import os
import sys
from pathlib import Path

from kedro.framework.project import configure_project
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# Change to project directory so relative paths resolve correctly
os.chdir(project_root)


def run_inference_pipeline(
    env: str = "local",
    pipeline_name: str = "inference",
) -> None:
    """Run the Kedro inference pipeline programmatically."""
    package_name = "ml_app_wind_draft"
    configure_project(package_name)
    bootstrap_project(project_root)

    with KedroSession.create(project_path=project_root, env=env) as session:
        session.run(pipeline_name=pipeline_name)


if __name__ == "__main__":
    run_inference_pipeline()
