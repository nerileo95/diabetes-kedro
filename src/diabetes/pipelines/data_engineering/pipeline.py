from kedro.pipeline import Node, Pipeline

from .nodes import add_split_column, clean_data, fit_imputer, transform_imputer


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=clean_data,
                inputs=["raw_diabetes_data", "params:columns"],
                outputs="cleaned_diabetes_data",
                name="clean_data",
            ),
            Node(
                func=add_split_column,
                inputs=["cleaned_diabetes_data", "params:split"],
                outputs="split_diabetes_data",
                name="add_split_column",
            ),
            Node(
                  func=fit_imputer,
                  inputs=["split_diabetes_data", "params:columns", "params:split_to_fit", "params:imputer"],
                  outputs="imputer",
                  name="fit_imputer",
            ),
            Node(
                func=transform_imputer,
                inputs=["split_diabetes_data", "imputer"],
                outputs="imputed_diabetes_data",
                name="transform_imputer",
            ),
        ]
    )