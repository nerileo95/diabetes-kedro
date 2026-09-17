import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from kedro.framework.project import pipelines
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from kedro.io import MemoryDataset
from kedro.runner import SequentialRunner
from pydantic import BaseModel

PROJECT_PATH = Path(__file__).resolve().parents[2]

_bootstrap_lock = threading.Lock()
_bootstrapped = False


def _ensure_bootstrap() -> None:
    global _bootstrapped
    if _bootstrapped:
        return
    with _bootstrap_lock:
        if not _bootstrapped:
            bootstrap_project(PROJECT_PATH)
            _bootstrapped = True

class Patient(BaseModel):
      Pregnancies: int
      Glucose: float
      BloodPressure: float
      SkinThickness: float
      Insulin: float
      BMI: float
      DiabetesPedigreeFunction: float
      Age: int


class InferenceRequest(BaseModel):
    instances: list[Patient]


app = FastAPI(title="Diabetes API")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

@app.post("/inference")
def inference(request: InferenceRequest) -> dict[str, Any]:
    _ensure_bootstrap()
    try:
        with KedroSession.create(project_path=PROJECT_PATH) as session:
            catalog = session.load_context().catalog
            catalog["raw_inference_data"] = MemoryDataset(
                [p.model_dump() for p in request.instances]
            )
            catalog["scaled_inference_data"] = MemoryDataset()
            catalog["inference_predictions"] = MemoryDataset()
            SequentialRunner().run(pipelines["inference"], catalog)
            return {"predictions": catalog.load("inference_predictions")}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.post("/batch-inference")
def batch_inference() -> dict[str, Any]:
    _ensure_bootstrap()
    with KedroSession.create(project_path=PROJECT_PATH) as session:
        session.run(pipeline_name="inference")
    return {"status": "completed", "output": "data/07_model_output/inference_predictions.json"}