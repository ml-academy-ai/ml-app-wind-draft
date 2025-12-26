"""Project pipelines."""

from __future__ import annotations

from kedro.pipeline import Pipeline

from ml_app_wind_draft.pipelines.feature_eng.pipeline import (
    feat_eng_pipeline_inference,
    feat_eng_pipeline_training,
)
from ml_app_wind_draft.pipelines.inference.pipeline import (
    create_pipeline as create_inference_pipeline,
)
from ml_app_wind_draft.pipelines.training.pipeline import (
    create_pipeline as create_training_pipeline,
)


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    feature_eng_pipeline_train = feat_eng_pipeline_training()
    feature_eng_pipeline_inf = feat_eng_pipeline_inference()
    training_pipeline = create_training_pipeline()
    inference_pipeline = create_inference_pipeline()

    return {
        "__default__": feature_eng_pipeline_train + training_pipeline,
        "training": feature_eng_pipeline_train + training_pipeline,
        "inference": feature_eng_pipeline_inf + inference_pipeline,
    }
