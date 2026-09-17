import logging
from typing import Any

import numpy as np
import pandas as pd

from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler

logger = logging.getLogger(__name__)


def clean_data(raw_data: pd.DataFrame, columns: dict[str, Any]) -> pd.DataFrame:
      """Seleciona as colunas conhecidas e troca zeros impossíveis por NaN."""
      wanted = columns["numerical"] + [columns["target"]]
      df = raw_data[[c for c in wanted if c in raw_data.columns]].copy()

      for col in columns["zero_as_missing"]:
          df[col] = df[col].replace(0, np.nan)

      logger.info("Cleaned data: %d rows, %d NaNs", len(df), df.isna().sum().sum())
      return df


def add_split_column(df: pd.DataFrame, split: dict[str, Any]) -> pd.DataFrame:
      """Sorteia train/test/validate para cada linha."""
      probs = [split["train"], split["test"], split["validate"]]
      if not np.isclose(sum(probs), 1.0):
          raise ValueError(f"Split proportions must sum to 1.0, got {sum(probs)}")

      rng = np.random.default_rng(split["random_state"])
      labels = rng.choice(["train", "test", "validate"], size=len(df), p=probs)
      return df.assign(split=labels)

def fit_imputer(
      df: pd.DataFrame, columns: dict[str, Any], split_to_fit: list[str], params: dict[str, Any]
  ) -> dict[str, Any]:
      """RobustScaler + KNNImputer (como no notebook), ajustados só no treino."""
      cols = columns["zero_as_missing"]
      train = df.loc[df["split"].isin(split_to_fit), cols]

      scaler = RobustScaler().fit(train)
      imputer = KNNImputer(n_neighbors=params["n_neighbors"]).fit(scaler.transform(train))
      return {"columns": cols, "scaler": scaler, "imputer": imputer}


def transform_imputer(df: pd.DataFrame, imputer: dict[str, Any]) -> pd.DataFrame:
    df_out = df.copy()
    cols = imputer["columns"]
    scaled = imputer["scaler"].transform(df_out[cols])
    filled = imputer["imputer"].transform(scaled)
    df_out[cols] = imputer["scaler"].inverse_transform(filled)
    return df_out

def fit_outlier_thresholds(
      df: pd.DataFrame, columns: dict[str, Any], split_to_fit: list[str], params: dict[str, Any]
  ) -> dict[str, list[float]]:
      """Limites de outlier (quantis 5%/95% + 1.5 IQR) calculados só no treino."""
      train = df.loc[df["split"].isin(split_to_fit)]
      thresholds = {}
      for col in columns["numerical"]:
          q1 = train[col].quantile(params["q1"])
          q3 = train[col].quantile(params["q3"])
          iqr = q3 - q1
          thresholds[col] = [float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)]
      return thresholds


def cap_outliers(df: pd.DataFrame, thresholds: dict[str, list[float]]) -> pd.DataFrame:
    df_out = df.copy()
    for col, (low, up) in thresholds.items():
        df_out[col] = df_out[col].clip(low, up)
    return df_out

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features do notebook. Não aprende nada dos dados, então não precisa de fit."""
    df = df.copy()
    age, bmi, glucose = df["Age"], df["BMI"], df["Glucose"]

    df["NEW_AGE_CAT"] = np.where(age >= 50, "senior", "mature")
    df["NEW_BMI"] = pd.cut(
        bmi, bins=[0, 18.5, 24.9, 29.9, 100], labels=["Underweight", "Healthy", "Overweight", "Obese"]
    ).astype(str)
    df["NEW_GLUCOSE"] = pd.cut(
        glucose, bins=[0, 140, 200, 300], labels=["Normal", "Prediabetes", "Diabetes"]
    ).astype(str)

    bmi_group = pd.cut(
        bmi, bins=[0, 18.5, 25, 30, 100], right=False,
        labels=["underweight", "healthy", "overweight", "obese"],
    ).astype(str)
    df["NEW_AGE_BMI_NOM"] = bmi_group + df["NEW_AGE_CAT"]

    glucose_group = pd.cut(
        glucose, bins=[0, 70, 100, 126, 1000], right=False,
        labels=["low", "normal", "hidden", "high"],
    ).astype(str)
    df["NEW_AGE_GLUCOSE_NOM"] = glucose_group + df["NEW_AGE_CAT"]

    df["NEW_INSULIN_SCORE"] = np.where(df["Insulin"].between(16, 166), "Normal", "Abnormal")
    df["NEW_GLUCOSE_X_INSULIN"] = glucose * df["Insulin"]
    df["NEW_GLUCOSE_X_PREGNANCIES"] = glucose * df["Pregnancies"]
    return df

def fit_encoders(
      df: pd.DataFrame, columns: dict[str, Any], split_to_fit: list[str]
  ) -> OneHotEncoder:
      cols = columns["engineered_categorical"]
      train = df.loc[df["split"].isin(split_to_fit), cols].astype(str)
      encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
      return encoder.set_output(transform="pandas").fit(train)


def transform_encoders(df: pd.DataFrame, encoder: OneHotEncoder) -> pd.DataFrame:
    cols = list(encoder.feature_names_in_)
    encoded = encoder.transform(df[cols].astype(str))
    return pd.concat([df.drop(columns=cols), encoded.set_index(df.index)], axis=1)

def fit_scalers(
      df: pd.DataFrame, columns: dict[str, Any], split_to_fit: list[str]
  ) -> RobustScaler:
      cols = columns["numerical"] + columns["engineered_numerical"]
      train = df.loc[df["split"].isin(split_to_fit), cols]
      return RobustScaler().set_output(transform="pandas").fit(train)


def transform_scalers(df: pd.DataFrame, scaler: RobustScaler) -> pd.DataFrame:
    df_out = df.copy()
    cols = list(scaler.feature_names_in_)
    df_out[cols] = scaler.transform(df_out[cols])
    return df_out