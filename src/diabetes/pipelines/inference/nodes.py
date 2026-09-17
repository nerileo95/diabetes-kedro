import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def to_dataframe(raw_data: pd.DataFrame | list[dict[str, Any]]) -> pd.DataFrame:
    """Aceita DataFrame (CSV do catálogo) ou lista de dicts (JSON da API)."""
    return raw_data if isinstance(raw_data, pd.DataFrame) else pd.DataFrame(raw_data)


def predict(model: dict[str, Any], data: pd.DataFrame) -> list[dict[str, Any]]:
    X = data[model["feature_columns"]]
    preds = model["estimator"].predict(X)
    probas = model["estimator"].predict_proba(X)[:, 1]
    logger.info("Generated %d predictions", len(preds))
    return [
        {"index": i, "prediction": int(p), "probability": float(pr)}
        for i, (p, pr) in enumerate(zip(preds, probas))
    ]
