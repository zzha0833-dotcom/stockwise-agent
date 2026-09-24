from __future__ import annotations

import json
from pathlib import Path

import typer

from stockwise.config import get_settings
from stockwise.data import (
    download_m5,
    generate_synthetic_retail_data,
    prepare_m5_subset,
    write_dataset,
)
from stockwise.evaluation import evaluate_dataset
from stockwise.schemas import ApprovalRequest, RunRequest, RunStatus
from stockwise.service import StockWiseService

app = typer.Typer(help="StockWise agent command line interface", no_args_is_help=True)


@app.command("generate-demo")
def generate_demo(
    items: int = typer.Option(8, min=1, max=100),
    stores: int = typer.Option(2, min=1, max=3),
    seed: int = 42,
) -> None:
    settings = get_settings()
    service = StockWiseService(settings)
    path = settings.data_dir / "processed" / f"synthetic_{items}x{stores}.parquet"
    profile = write_dataset(
        generate_synthetic_retail_data(items=items, stores=stores, seed=seed), path
    )
    record = service.storage.register_dataset(
        name=f"Synthetic retail {items}x{stores}", path=path, profile=profile
    )
    typer.echo(json.dumps({"dataset_id": record.id, "profile": profile.model_dump()}, indent=2))


@app.command("download-m5")
def download_m5_command(
    accept_source_terms: bool = typer.Option(
        False, "--accept-source-terms", help="Confirm that you reviewed the upstream terms"
    ),
) -> None:
    settings = get_settings()
    files = download_m5(settings.data_dir / "raw" / "m5", accept_source_terms=accept_source_terms)
    typer.echo("\n".join(str(path) for path in files))


@app.command("prepare-m5")
def prepare_m5_command(
    source_dir: Path = typer.Option(..., exists=True, file_okay=False),
    items: int = typer.Option(30, min=1, max=100),
    stores: str = typer.Option("CA_1,TX_1,WI_1"),
) -> None:
    settings = get_settings()
    service = StockWiseService(settings)
    path = settings.data_dir / "processed" / f"m5_{items}items.parquet"
    profile = prepare_m5_subset(
        source_dir, path, item_limit=items, stores=[item.strip() for item in stores.split(",")]
    )
    record = service.storage.register_dataset(name=f"M5 {items} item subset", path=path, profile=profile)
    typer.echo(json.dumps({"dataset_id": record.id, "profile": profile.model_dump()}, indent=2))


@app.command("run-demo")
def run_demo(
    items: int = typer.Option(8, min=1, max=30),
    auto_approve: bool = typer.Option(True, "--auto-approve/--no-auto-approve"),
) -> None:
    settings = get_settings()
    settings.sync_runs = True
    service = StockWiseService(settings)
    dataset = service.ensure_demo_dataset(items=max(items, 8), stores=2)
    run = service.submit_run(
        RunRequest(
            dataset_id=dataset.id,
            item_limit=items,
            horizon=28,
            experiment_budget=3,
            approval_amount_threshold=5000,
        ),
        synchronous=True,
    )
    run = service.storage.get_run(run.id) or run
    if run.status == RunStatus.AWAITING_APPROVAL.value and auto_approve:
        service.approve(
            run.id, ApprovalRequest(decision="approve", note="CLI auto approval"), synchronous=True
        )
        run = service.storage.get_run(run.id) or run
    typer.echo(json.dumps({"run_id": run.id, "status": run.status, "error": run.error}, indent=2))


@app.command("serve-api")
def serve_api(host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> None:
    import uvicorn

    uvicorn.run("stockwise.api:app", host=host, port=port, reload=reload)


@app.command("evaluate-demo")
def evaluate_demo(
    items: int = typer.Option(30, min=1, max=100),
    runs: int = typer.Option(3, min=1, max=5),
) -> None:
    settings = get_settings()
    settings.sync_runs = True
    service = StockWiseService(settings)
    dataset = service.ensure_demo_dataset(items=items, stores=3)
    result = evaluate_dataset(
        service,
        dataset_id=dataset.id,
        item_limit=items,
        seeds=list(range(42, 42 + runs)),
        output_dir=settings.artifact_dir / "evaluations",
        dataset_label="synthetic",
    )
    typer.echo(json.dumps(result["aggregate"], indent=2))
    typer.echo(result["html_path"])


@app.command("evaluate-m5")
def evaluate_m5(
    items: int = typer.Option(30, min=1, max=100),
    runs: int = typer.Option(3, min=1, max=5),
) -> None:
    """Evaluate a prepared M5 subset with reproducible seeds."""
    settings = get_settings()
    settings.sync_runs = True
    service = StockWiseService(settings)
    expected_name = f"M5 {items} item subset"
    dataset = next(
        (record for record in service.storage.list_datasets() if record.name == expected_name),
        None,
    )
    if dataset is None:
        raise typer.BadParameter(
            f"No '{expected_name}' dataset is registered. Run prepare-m5 first."
        )
    result = evaluate_dataset(
        service,
        dataset_id=dataset.id,
        item_limit=items,
        seeds=list(range(42, 42 + runs)),
        output_dir=settings.artifact_dir / "evaluations",
        dataset_label="m5",
    )
    typer.echo(json.dumps(result["aggregate"], indent=2))
    typer.echo(result["html_path"])


if __name__ == "__main__":
    app()
