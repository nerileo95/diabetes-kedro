import importlib
import logging
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV

logger = logging.getLogger(__name__)


def _load_class(class_path: str) -> type:
    module_name, class_name = class_path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), class_name)


def _feature_columns(master_table: pd.DataFrame, target: str) -> list[str]:
    return [c for c in master_table.columns if c not in (target, "split")]

def train_model(
    master_table: pd.DataFrame, columns: dict[str, Any], params: dict[str, Any]
) -> dict[str, Any]:
    target = columns["target"]
    features = _feature_columns(master_table, target)
    train = master_table[master_table["split"].isin(params["train_splits"])]

    estimator = _load_class(params["class_path"])(**params.get("init_args", {}))
    estimator.fit(train[features], train[target])
    logger.info("Trained %s on %d rows", params["class_path"], len(train))

    return {
        "estimator": estimator,
        "target_column": target,
        "feature_columns": features,
        "eval_splits": params["eval_splits"],
    }

def evaluate_model(model: dict[str, Any], master_table: pd.DataFrame) -> dict[str, Any]:
    metrics = {}
    for split in model["eval_splits"]:
        df = master_table[master_table["split"] == split]
        y_true = df[model["target_column"]]
        X = df[model["feature_columns"]]
        y_pred = model["estimator"].predict(X)
        y_proba = model["estimator"].predict_proba(X)[:, 1]
        metrics[split] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred)),
            "recall": float(recall_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred)),
            "roc_auc": float(roc_auc_score(y_true, y_proba)),
            "n_samples": len(df),
        }
        logger.info("%s: %s", split, metrics[split])
    return metrics

def optimize_hyperparameters(
      master_table: pd.DataFrame, columns: dict[str, Any], params: dict[str, Any]
  ) -> dict[str, Any]:
      target = columns["target"]
      features = _feature_columns(master_table, target)
      train = master_table[master_table["split"].isin(params["train_splits"])]

      base = _load_class(params["class_path"])(**params.get("init_args", {}))
      search = GridSearchCV(
          base, params["param_grid"], cv=params["cv"], scoring=params["scoring"], n_jobs=-1
      )
      search.fit(train[features], train[target])
      logger.info("Best %s=%.4f with %s", params["scoring"], search.best_score_, search.best_params_)

      return {
          "estimator": search.best_estimator_,
          "target_column": target,
          "feature_columns": features,
          "eval_splits": params["eval_splits"],
          "best_params": search.best_params_,
      }
