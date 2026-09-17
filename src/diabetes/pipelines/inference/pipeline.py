from kedro.pipeline import Node, Pipeline

from diabetes.pipelines.data_engineering.nodes import (
    cap_outliers,
    clean_data,
    create_features,
    transform_encoders,
    transform_imputer,
    transform_scalers,
)

from .nodes import predict, to_dataframe


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(to_dataframe, "raw_inference_data", "raw_inference_dataframe", name="to_dataframe"),
            Node(clean_data, ["raw_inference_dataframe", "params:columns"], "cleaned_inference_data", name="clean_inference_data"),
            Node(transform_imputer, ["cleaned_inference_data", "imputer"], "imputed_inference_data", name="impute_inference_data"),
            Node(cap_outliers, ["imputed_inference_data", "outlier_thresholds"], "capped_inference_data", name="cap_inference_outliers"),
            Node(create_features, "capped_inference_data", "featured_inference_data", name="create_inference_features"),
            Node(transform_encoders, ["featured_inference_data", "encoders"], "encoded_inference_data", name="encode_inference_data"),
            Node(transform_scalers, ["encoded_inference_data", "scalers"], "scaled_inference_data", name="scale_inference_data"),
            Node(predict, ["optimized_model", "scaled_inference_data"], "inference_predictions", name="predict"),
        ]
    )
