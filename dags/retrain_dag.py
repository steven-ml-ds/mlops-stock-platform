"""Weekly retrain pipeline: fetch fresh data, train a challenger, let it
fight the champion for the @production alias.

Each task is a thin wrapper around platform_core so the logic stays
testable outside Airflow; the DAG only handles orchestration and XCom.
"""

from datetime import datetime

from airflow.sdk import dag, task

WIN_PROB_THRESHOLD = 0.60


@dag(
    dag_id="weekly_retrain",
    schedule="@weekly",
    start_date=datetime(2026, 6, 1),
    catchup=False,
    tags=["mlops", "retrain"],
)
def weekly_retrain():
    @task
    def fetch_data() -> int:
        from platform_core.data import fetch_all

        frames = fetch_all(refresh=True)
        return sum(len(f) for f in frames.values())

    @task
    def check_production_decay(n_rows: int) -> dict:
        """Informational gate: logs recent production accuracy to MLflow.
        The weekly retrain proceeds regardless; this trends the decay signal."""
        from platform_core.monitor import check_decay

        try:
            return check_decay()
        except Exception as exc:  # cold start: no production model yet
            return {"degraded": None, "note": str(exc)}

    @task
    def train_challenger(decay: dict) -> str:
        from platform_core.promote import DEFAULT_PARAMS
        from platform_core.train import train_once

        return train_once(DEFAULT_PARAMS, run_name="weekly-challenger", register=True)

    @task
    def promote_if_better(challenger_run_id: str) -> dict:
        from platform_core.promote import compare_and_promote

        return compare_and_promote(challenger_run_id, threshold=WIN_PROB_THRESHOLD)

    promote_if_better(train_challenger(check_production_decay(fetch_data())))


weekly_retrain()
