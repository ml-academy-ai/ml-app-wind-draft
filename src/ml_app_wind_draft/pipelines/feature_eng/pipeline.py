from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    add_lag_features,
    add_rolling_features,
    drop_columns,
    get_data_timestamps,
    get_features_and_target,
    get_last_n_points_from_db,
    load_data,
    load_training_data_from_db,
    remove_diff_outliers,
    rename_columns,
    smooth_signal,
)

# def feat_eng_pipeline_training(**kwargs) -> Pipeline:
#     return load_training_data() + create_feat_eng_pipeline()

# def feat_eng_pipeline_inference(**kwargs) -> Pipeline:
#     return load_inference_data() + create_feat_eng_pipeline()


def feat_eng_pipeline_training(**kwargs) -> Pipeline:
    return load_training_data_from_database() + create_feat_eng_pipeline()


def feat_eng_pipeline_inference(**kwargs) -> Pipeline:
    return load_inference_data_from_db() + create_feat_eng_pipeline()


def load_training_data(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=load_data,
                inputs="train_df",
                outputs="loaded_df",
            ),
        ]
    )


def load_training_data_from_database(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=load_training_data_from_db,
                inputs=[
                    "params:training_pipeline.start_timestamp",
                    "params:training_pipeline.end_timestamp",
                    "params:data_manager.raw_data_table_name",
                    "params:data_manager",
                ],
                outputs="loaded_df",
            ),
        ]
    )


def load_inference_data(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=load_data,
                inputs="inference_df",
                outputs="loaded_df",
            ),
        ]
    )


def load_inference_data_from_db(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=get_last_n_points_from_db,
                inputs=[
                    "params:inference_pipeline.batch_size",
                    "params:data_manager.raw_data_table_name",
                    "params:data_manager",
                ],
                outputs="loaded_df",
            ),
        ]
    )


def create_feat_eng_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=rename_columns,
                inputs=[
                    "loaded_df",
                    "params:feature_eng_pipeline.rename_columns",
                ],
                outputs="renamed_data",
            ),
            node(
                func=get_data_timestamps,
                inputs="renamed_data",
                outputs="data_timestamps",
            ),
            node(
                func=drop_columns,
                inputs=[
                    "renamed_data",
                    "params:feature_eng_pipeline.drop_columns",
                ],
                outputs="dropped_columns_data",
            ),
            node(
                func=remove_diff_outliers,
                inputs=[
                    "dropped_columns_data",
                    "params:feature_eng_pipeline.diff_outliers_thresholds",
                ],
                outputs="removed_outliers_data",
            ),
            node(
                func=smooth_signal,
                inputs=[
                    "removed_outliers_data",
                    "params:feature_eng_pipeline.smooth_signal.columns",
                    "params:feature_eng_pipeline.smooth_signal.window",
                    "params:feature_eng_pipeline.smooth_signal.method",
                ],
                outputs="smoothed_data",
            ),
            node(
                func=add_lag_features,
                inputs=[
                    "smoothed_data",
                    "params:feature_eng_pipeline.lag_features",
                ],
                outputs="lagged_data",
            ),
            node(
                func=add_rolling_features,
                inputs=[
                    "lagged_data",
                    "params:feature_eng_pipeline.rolling_features",
                ],
                outputs="rolled_stats_data",
            ),
            node(
                func=get_features_and_target,
                inputs=[
                    "rolled_stats_data",
                    "params:feature_eng_pipeline",
                ],
                outputs=["features_data", "target_data"],
            ),
        ]
    )
