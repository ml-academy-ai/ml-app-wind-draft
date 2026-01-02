import os
import sys
import tomllib
from datetime import timedelta
from pathlib import Path

from kedro.framework.project import configure_project
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from prefect import flow, task

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
os.chdir(project_root)


@task(name="training-task")
def training_task(env: str = "local", pipeline_name: str = "training"):
    """Prefect task wrapper."""
    # Extract package name from pyproject.toml
    with open(project_root / "pyproject.toml", "rb") as f:
        package_name = tomllib.load(f)["tool"]["kedro"]["package_name"]

    configure_project(package_name)
    bootstrap_project(project_root)

    with KedroSession.create(project_path=project_root, env=env) as session:
        session.run(pipeline_name=pipeline_name)


@flow(name="training-flow")
def training_flow(env: str = "local"):
    """Prefect flow for training."""
    training_task(env=env)


if __name__ == "__main__":
    os.environ.setdefault("PREFECT_API_URL", "http://127.0.0.1:4200/api")

    training_flow.serve(
        name="training-flow",
        interval=timedelta(minutes=1),
        parameters={"env": os.getenv("KEDRO_ENV", "local")},
    )

# if __name__ == "__main__":
#     os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
#     training_flow(env="local")
