from kedro.pipeline import Pipeline, node

from .nodes import (
    compute_metrics,
    load_from_registry,
    predict,
    save_predictions_to_db,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=load_from_registry,
                inputs=["params:mlflow.registered_model_name"],
                outputs="best_model",
            ),
            node(
                func=predict,
                inputs=["features_data", "best_model"],
                outputs="y_pred",
            ),
            node(
                func=compute_metrics,
                inputs=["target_data", "y_pred"],
                outputs="metrics",
            ),
            node(
                func=save_predictions_to_db,
                inputs=[
                    "y_pred",
                    "data_timestamps",
                    "params:data_manager",
                ],
                outputs=None,
            ),
        ]
    )
