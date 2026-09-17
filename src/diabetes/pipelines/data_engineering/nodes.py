import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def clean_data(raw_data: pd.DataFrame, columns: dict[str, Any]) -> pd.DataFrame:
      """Seleciona as colunas conhecidas e troca zeros impossíveis por NaN."""
      wanted = columns["numerical"] + [columns["target"]]
      df = raw_data[[c for c in wanted if c in raw_data.columns]].copy()

      for col in columns["zero_as_missing"]:
          df[col] = df[col].replace(0, np.nan)

      logger.info("Cleaned data: %d rows, %d NaNs", len(df), df.isna().sum().sum())
      return df