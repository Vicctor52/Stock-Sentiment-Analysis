# Financial Sentiment Lab

Pipeline de analisis de sentimiento financiero que combina noticias y precios
bursatiles para estudiar relaciones lead-lag entre las noticias y los retornos.

## Arquitectura

```mermaid
flowchart LR
    Y[yfinance] --> ING[Ingesta]
    N[NewsAPI] --> ING
    ING --> NRM[Normalizacion UTC + sesion NYSE]
    NRM --> PQ[(Parquet)]
    PQ --> NLP[Limpieza + FinBERT]
    NLP --> DS[(Sentimiento diario)]
    PQ --> AN[Retornos + lead-lag + Granger]
    DS --> AN
    AN --> RES[(Resultados)]
    RES --> DASH[Streamlit + Plotly]
```

## Estructura

```text
finsent/
  ingestion/   clientes de noticias y precios
  storage/     esquemas y persistencia Parquet
  nlp/         limpieza, FinBERT y agregacion diaria
  analysis/    alineacion, retornos, lead-lag y Granger
  pipeline/    orquestadores
  dashboard/   aplicacion Streamlit
config/        tickers y parametros
data/          raw, processed, cache y sample
tests/         pruebas unitarias y sinteticas
```

## Requisitos

- Python 3.12
- [uv](https://docs.astral.sh/uv/)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12
uv pip install -r requirements-pipeline.txt
```

## Uso

```bash
uv run python -m finsent ingest
uv run python -m finsent score
uv run python -m finsent analyze
uv run python -m finsent daily
uv run streamlit run streamlit_app.py
```

`daily` procesa las sesiones NYSE cerradas que falten desde el último checkpoint:
descarga noticias entre cierres, actualiza precios, puntúa solo artículos nuevos y
recalcula el análisis. El workflow de GitHub Actions lo ejecuta de lunes a viernes a
las 22:30 UTC y guarda los Parquet y el checkpoint en el repositorio. La primera
ejecución descarga el histórico de precios configurado y las noticias del intervalo
más reciente; las siguientes son incrementales. Configura `NEWSAPI_KEY` como secreto
del repositorio y permite escritura para GitHub Actions en **Settings → Actions →
General → Workflow permissions**. Los Parquet se guardan mediante commits, por lo que
el historial del repositorio crecerá con el tiempo.

El dashboard lee los Parquet de `data/processed` y muestra el sentimiento y los
retornos diarios, además de los resultados lead-lag y Granger. Usa **Recargar datos**
en la barra lateral tras regenerar los resultados. Si faltan tablas, la app indica
qué etapas del pipeline debes ejecutar.

## Variables de entorno

Copia `.env.example` a `.env` y completa las credenciales.

```bash
cp .env.example .env
```

## Estado

Proyecto centrado exclusivamente en noticias y precios. Las fuentes sociales y los
formularios regulatorios quedan fuera del alcance. La ingesta de precios y noticias, el
scoring FinBERT, la agregacion diaria, el analisis de retornos, lead-lag y Granger estan
implementados y conectados al dashboard.
