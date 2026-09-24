# Predição de Diabetes: Pipeline Kedro

Projeto da disciplina Deploy de Data Science (Insper). Reescreve o notebook
`diabetes-prediction.ipynb` como um projeto Kedro, com API e container.

**Integrantes:** Leonardo Caraça; Alisson Moriyasu; Cristiane Gonçalves

## Pipelines
- `data_engineering` (11 nós): limpeza, split, imputação KNN, outliers, features, encoding e scaling → `master_table`
- `modelling` (4 nós): baseline (regressão logística) e RandomForest com GridSearch, mais as métricas
- `inference` (8 nós): reusa os transforms e os artefatos do treino para prever, sem fazer nenhum fit

## Como rodar
```bash
uv sync                  # instala as dependências
uv run kedro run         # roda os 3 pipelines
uv run kedro viz run     # visualiza em http://127.0.0.1:4141
```

## API
```bash
uv run uvicorn diabetes.api:app --reload --port 8000
```
Documentação em http://127.0.0.1:8000/docs. Endpoints: `/health`, `/inference`, `/batch-inference`.

## Docker
```bash
docker compose up --build    # API em http://localhost:8001
```
O container usa a porta **8001** no host para não conflitar com o uvicorn local na 8000.

## Resultados (roc_auc)
| Modelo | Train | Test | Validate |
|---|---|---|---|
| Regressão logística | 0.874 | 0.726 | 0.897 |
| RandomForest otimizado | 0.987 | 0.725 | 0.880 |

O RandomForest mostra overfitting (0.987 no train contra 0.725 no test), por isso o baseline é o modelo preferido.
