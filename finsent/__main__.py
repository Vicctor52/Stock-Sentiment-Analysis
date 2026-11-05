import typer

from finsent.pipeline.analyze import run_analysis
from finsent.pipeline.ingest import run_ingestion
from finsent.pipeline.score import run_scoring

app = typer.Typer(help="Financial Sentiment Lab pipeline.")


@app.command()
def ingest() -> None:
    """Run the ingestion stage."""
    run_ingestion()


@app.command()
def score() -> None:
    """Run the sentiment scoring stage."""
    run_scoring()


@app.command()
def analyze() -> None:
    """Run the analysis stage."""
    run_analysis()


if __name__ == "__main__":
    app()
